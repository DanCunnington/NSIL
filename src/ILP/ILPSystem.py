from os.path import exists, dirname, abspath, join
from global_config import ILP_CMD_LINE_ARGS, NUM_CPU
import tempfile
import time
import subprocess

BASE_DIR = dirname(dirname(dirname(abspath(__file__))))


class ILP:
    def __init__(self, args, logger):
        """
        Class to represent the ILP System
        """
        self.args = args
        self.logger = logger

    def run(self, task, iteration):
        """
        Run the ILP system on the given task
        @param task: the ILP LAS task to run
        @param iteration: the iteration number
        @return: (learned hypothesis, learning time)
        """
        temp_task = tempfile.NamedTemporaryFile(prefix='NSIL_tmp_file_')
        temp_task.write(task.encode())

        # Build command line arguments
        if self.args.custom_ilp_cmd_line_args is None:
            cmd_line_args = ILP_CMD_LINE_ARGS[self.args.ilp_system]
        else:
            cmd_line_args = self.args.custom_ilp_cmd_line_args

        # Add pylasp script path
        if self.args.pylasp:
            assert self.args.ilp_system == 'ILASP'
            pylasp_path = join(self.logger.pylasp_dir, f'iteration_{iteration}.las')
            cmd_line_args = cmd_line_args.replace('--version=4', pylasp_path)

        # Add num threads to FastLAS calls
        if self.args.ilp_system == 'FastLAS':
            cmd_line_args += f' --threads {NUM_CPU}'

            # Read or write cache depending if the cache exists
            cp = join(self.logger.las_cache_dir, 'fastlas_cache')
            if exists(cp):
                cmd_line_args += f' --read-cache {cp}'
            else:
                cmd_line_args += f' --write-cache {cp}'

        # Remove other output from ILASP
        if self.args.ilp_system == 'ILASP':
            cmd_line_args += ' --quiet'

        cmd = f'{join(BASE_DIR, self.args.ilp_system)} {temp_task.name} {cmd_line_args}'
        self.logger.info('start_ilp', cmd)
        ilp_start_time = time.time()
        result = subprocess.run(cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                shell=True)

        output = result.stdout.decode()
        ilp_time = time.time() - ilp_start_time

        # Parse the output
        output = "\n".join([ll.rstrip() for ll in output.splitlines() if ll.strip() and ll != ""])

        # Close temporary file
        temp_task.close()
        self.logger.info('new_h', output)
        return output, ilp_time
