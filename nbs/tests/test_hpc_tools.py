import pytest
from pathlib import Path
import tempfile
import os
from typing import List, Callable, Any
import subprocess
from unittest.mock import patch, MagicMock

# Mock function for testing
def mock_process_func(input_path: str, output_path: str) -> bool:
    return True

def test_create_job_script():
    from cv_tools.preprocessing.hpc_tools import create_job_script
    
    with tempfile.TemporaryDirectory() as tmpdir:
        job_script = create_job_script(
            function_name="process_images",
            input_path="/path/to/input",
            output_path="/path/to/output",
            memory=16000,
            queue="normal",
            python_path="/usr/bin/python"
        )
        
        assert "#!/bin/bash" in job_script
        assert "#BSUB -M 16000" in job_script
        assert "#BSUB -q normal" in job_script
        assert "python -c" in job_script

def test_batch_generator():
    from cv_tools.preprocessing.hpc_tools import create_batches
    
    files = [f"file_{i}.jpg" for i in range(100)]
    batch_size = 10
    batches = list(create_batches(files, batch_size))
    
    assert len(batches) == 10
    assert len(batches[0]) == batch_size
    assert batches[0][0] == "file_0.jpg"

def test_submit_job():
    from cv_tools.preprocessing.hpc_tools import submit_job
    
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        
        success = submit_job("test_job.sh")
        assert success
        mock_run.assert_called_once()

def test_parallel_job_submission():
    from cv_tools.preprocessing.hpc_tools import submit_parallel_jobs
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_files = [f"{tmpdir}/input_{i}.jpg" for i in range(10)]
        output_dir = f"{tmpdir}/output"
        
        # Create dummy input files
        for f in input_files:
            Path(f).touch()
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            
            results = submit_parallel_jobs(
                function=mock_process_func,
                input_files=input_files,
                output_dir=output_dir,
                batch_size=2,
                memory=16000,
                queue="normal"
            )
            
            assert len(results) == 5  # 10 files / 2 batch_size = 5 jobs
            assert all(results)  # All jobs should be submitted successfully

def test_job_monitoring():
    from cv_tools.preprocessing.hpc_tools import monitor_jobs
    
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="JOBID   USER    STAT  QUEUE      FROM_HOST   EXEC_HOST   JOB_NAME\n123     user    RUN   normal     host1       host2       job1"
        )
        
        job_ids = ["123"]
        status = monitor_jobs(job_ids)
        assert status["123"] == "RUN" 