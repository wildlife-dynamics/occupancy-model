#!/bin/bash

set -e  # Exit on error

print_help() {
    echo "Usage: $0 <--all | --case test_case_name> [--update | --frozen] [--quiet|-q]"
    echo ""
    echo "Run from a workflow repo root. The script auto-discovers the inner"
    echo "ecoscope-workflows-*-workflow/ package and uses its pixi manifest."
    echo ""
    echo "Examples:"
    echo "  $0 --case base                 # Run single test case (default: pixi run --locked)"
    echo "  $0 --all                       # Run all test cases"
    echo "  $0 --all --quiet               # Minimal output"
    echo "  $0 --case base --update        # pixi update first, then --locked"
    echo "  $0 --case base --frozen        # pixi run --frozen (skip lockfile check)"
    echo ""
    echo "Options:"
    echo "  --case <name>   Run a specific test case"
    echo "  --all           Run all test cases for the workflow"
    echo "  --update        Run 'pixi update' on the inner manifest first; pixi run uses --locked"
    echo "  --frozen        Use 'pixi run --frozen' (no update, skip lockfile check)"
    echo "  --local         Run commands directly without 'pixi run' wrapping"
    echo "  --quiet, -q     Minimal output: only show pass/fail and errors"
    echo "  -h, --help      Show this help message"
}

update=false
frozen=false
local_mode=false
run_all=false
quiet=false
test_case=""

# Check for flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --update)
            update=true
            shift
            ;;
        --frozen)
            frozen=true
            shift
            ;;
        --local)
            local_mode=true
            shift
            ;;
        --all)
            run_all=true
            shift
            ;;
        --quiet|-q)
            quiet=true
            shift
            ;;
        --case)
            test_case="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ "$run_all" = false ] && [ -z "$test_case" ]; then
    echo "ERROR: Must specify either --all or --case <test_case_name>"
    exit 1
fi

if [ "$run_all" = true ] && [ -n "$test_case" ]; then
    echo "ERROR: Cannot specify both --all and --case"
    exit 1
fi

if [ "$update" = true ] && [ "$frozen" = true ]; then
    echo "ERROR: Cannot specify both --update and --frozen"
    exit 1
fi

# Preflight: required tools
required=(yq)
[ "$local_mode" = false ] && required+=(pixi)
missing=()
for tool in "${required[@]}"; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if [ ${#missing[@]} -gt 0 ]; then
    echo "ERROR: required tool(s) not found on PATH: ${missing[*]}" >&2
    exit 1
fi

# Auto-discover the inner workflow package from the repo layout.
repo_root=$(pwd)
shopt -s nullglob
workflow_dirs=("${repo_root}"/ecoscope-workflows-*-workflow)
shopt -u nullglob
if [ ${#workflow_dirs[@]} -ne 1 ]; then
    echo "ERROR: expected exactly one ecoscope-workflows-*-workflow directory under $repo_root, found ${#workflow_dirs[@]}"
    exit 1
fi
workflow_dir="${workflow_dirs[0]}"
workflow_dash=$(basename "$workflow_dir" | sed 's/^ecoscope-workflows-//; s/-workflow$//')
manifest_path="${workflow_dir}/pixi.toml"
test_cases_file="${repo_root}/test-cases.yaml"

if [ "$local_mode" = true ]; then
    mode_str="local (no pixi run)"
else
    lock_str="--locked"
    [ "$frozen" = true ] && lock_str="--frozen"
    mode_str="pixi run $lock_str"
    [ "$update" = true ] && mode_str="$mode_str (after pixi update)"
fi

if [ "$quiet" = false ]; then
    echo "=========================================="
    echo "Workflow:  $workflow_dash"
    if [ "$run_all" = true ]; then
        echo "Running:   ALL test cases"
    else
        echo "Test case: $test_case"
    fi
    echo "Mode:      $mode_str"
    echo "=========================================="
fi

# Helper function to run commands with or without pixi
run_cmd() {
    if [ "$local_mode" = true ]; then
        "$@"
    else
        local lock_flag="--locked"
        [ "$frozen" = true ] && lock_flag="--frozen"
        pixi run --manifest-path "$manifest_path" "$lock_flag" -e default "$@"
    fi
}

# Optional pixi update
if [ "$update" = true ]; then
    [ "$quiet" = false ] && echo "Running pixi update on inner manifest..."
    pixi update --manifest-path "$manifest_path"
fi

# Function to run a single test case
run_single_test_case() {
    local test_case=$1

    if [ "$quiet" = false ]; then
        echo ""
        echo "=========================================="
        echo "Running test case: $test_case"
        echo "=========================================="
    fi

    # Verify test case exists
    if ! yq -e ".\"${test_case}\"" "$test_cases_file" > /dev/null 2>&1; then
        echo "✗ $test_case — ERROR: test case not found in $test_cases_file"
        return 1
    fi

    # Extract mock_io setting from test case (defaults to true if not specified)
    if yq -e ".\"${test_case}\" | has(\"mock_io\")" "$test_cases_file" > /dev/null 2>&1; then
        use_mock_io=$(yq -r ".\"${test_case}\".mock_io" "$test_cases_file")
    else
        use_mock_io="true"
    fi
    [ "$quiet" = false ] && echo "Mock IO mode: $use_mock_io"

    # Create temporary results directory (cross-platform compatible)
    # Use RUNNER_TEMP if available (GitHub Actions), otherwise fall back to /tmp
    temp_base="${RUNNER_TEMP:-/tmp}"
    results_dir="${temp_base}/workflow-test-results/${workflow_dash}/${test_case}"
    rm -rf "$results_dir"
    mkdir -p "$results_dir"
    [ "$quiet" = false ] && echo "Created results directory: $results_dir"
    [ "$quiet" = false ] && echo ""

    # Export ECOSCOPE_WORKFLOWS_RESULTS for workflow to use
    export ECOSCOPE_WORKFLOWS_RESULTS="file://${results_dir}"

    # Export mock_io_overrides as WT_TASK_MOCK_IO__* env vars.
    # Each key is a dotted task path (e.g. "pkg.tasks.module.func_name") which
    # is uppercased with dots → underscores, e.g.
    #   pkg.tasks.module.func_name → WT_TASK_MOCK_IO__PKG_TASKS_MODULE_FUNC_NAME
    local _mock_override_vars=()
    if yq -e ".\"${test_case}\" | has(\"mock_io_overrides\")" "$test_cases_file" > /dev/null 2>&1; then
        while IFS= read -r dotted_path; do
            file_path=$(yq -r ".\"${test_case}\".mock_io_overrides.\"${dotted_path}\"" "$test_cases_file")
            # Resolve repo-relative paths to absolute. Leave URLs and absolute
            # paths untouched. The mock loader requires absolute paths because
            # it calls Path.as_uri().
            case "$file_path" in
                /*|http://*|https://*|file://*) ;;
                *) file_path="${repo_root}/${file_path#./}" ;;
            esac
            env_name="WT_TASK_MOCK_IO__$(echo "$dotted_path" | tr '.' '_' | tr '[:lower:]' '[:upper:]')"
            export "$env_name"="$file_path"
            _mock_override_vars+=("$env_name")
            [ "$quiet" = false ] && echo "Mock override: $env_name=$file_path"
        done < <(yq -r ".\"${test_case}\".mock_io_overrides | keys | .[]" "$test_cases_file")
    fi

    # Extract params for this test case
    params_file="${results_dir}/params.yaml"
    yq ".\"${test_case}\".params" "$test_cases_file" > "$params_file"

    if [ "$quiet" = false ]; then
        echo "Extracted params:"
        cat "$params_file"
        echo ""
    fi

    # Run workflow CLI directly
    if [ "$quiet" = false ]; then
        echo "Executing workflow..."
        echo "Results will be written to: $ECOSCOPE_WORKFLOWS_RESULTS"
        echo ""
    fi

    cd "$workflow_dir"
    workflow_underscore="${workflow_dash//-/_}"

    cmd=(python -m "ecoscope_workflows_${workflow_underscore}_workflow.cli" run
         --config-file "$params_file" --execution-mode sequential)
    if [ "$use_mock_io" = "true" ]; then
        cmd+=(--mock-io)
    fi

    if [ "$quiet" = false ]; then
        echo "Command: ${cmd[*]}"
        echo ""
    fi

    # Run the command and capture exit code
    if [ "$quiet" = true ]; then
        if run_cmd "${cmd[@]}" > /dev/null 2>&1; then
            cmd_exit_code=0
        else
            cmd_exit_code=$?
        fi
    else
        if run_cmd "${cmd[@]}"; then
            cmd_exit_code=0
        else
            cmd_exit_code=$?
        fi
    fi

    # Return to repo root
    cd "$repo_root"

    # Clean up mock override env vars so they don't leak to the next test case
    for _var in "${_mock_override_vars[@]}"; do
        unset "$_var"
    done

    # Validate result.json
    result_json="${results_dir}/result.json"
    if [ ! -f "$result_json" ]; then
        echo "✗ $test_case — result.json not found at $result_json"
        return 1
    fi

    [ "$quiet" = false ] && echo ""
    [ "$quiet" = false ] && echo "Validating result.json..."
    error_value=$(yq -p json -r '.error // "null"' "$result_json")

    if [ "$error_value" != "null" ] || [ $cmd_exit_code -ne 0 ]; then
        echo "✗ $test_case — FAILED"
        if [ "$error_value" != "null" ]; then
            echo "  Error: $(yq -p json -r '.error' "$result_json")"
        fi
        [ "$quiet" = false ] && echo "" && echo "Full result.json:" && cat "$result_json"
        return 1
    fi

    echo "✓ $test_case — passed"
    if [ "$quiet" = false ]; then
        echo ""
        echo "Full result.json:"
        cat "$result_json"
    fi

    return 0
}

# Main logic: run all test cases or a single one
if [ "$run_all" = true ]; then
    # Get all test case names from test-cases.yaml
    # tr -d '\r' removes carriage returns for Windows compatibility
    test_cases=($(yq 'keys | .[]' "$test_cases_file" | tr -d '"\r'))

    if [ "$quiet" = false ]; then
        echo ""
        echo "Found ${#test_cases[@]} test cases: ${test_cases[*]}"
        echo ""
    fi

    # Track results
    declare -a failed_cases
    declare -a passed_cases

    # Loop through each test case
    for test_case in "${test_cases[@]}"; do
        if run_single_test_case "$test_case"; then
            passed_cases+=("$test_case")
        else
            failed_cases+=("$test_case")
        fi
    done

    # Print summary
    echo ""
    echo "=========================================="
    echo "TEST SUMMARY"
    echo "=========================================="
    echo "Total: ${#test_cases[@]}"
    echo "Passed: ${#passed_cases[@]}"
    echo "Failed: ${#failed_cases[@]}"
    echo ""

    if [ ${#passed_cases[@]} -gt 0 ]; then
        echo "✓ Passed test cases:"
        for case in "${passed_cases[@]}"; do
            echo "  - $case"
        done
        echo ""
    fi

    if [ ${#failed_cases[@]} -gt 0 ]; then
        echo "✗ Failed test cases:"
        for case in "${failed_cases[@]}"; do
            echo "  - $case"
        done
        echo ""
        exit 1
    fi

    echo "✓ All tests passed!"

else
    # Run single test case
    run_single_test_case "$test_case"
fi
