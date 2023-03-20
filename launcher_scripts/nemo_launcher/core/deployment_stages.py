# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
from pathlib import Path
from typing import Dict, List

from nemo_launcher.core.launchers import AutoLauncher
from nemo_launcher.core.stages import NemoMegatronStage, clean_command_groups

PYTHON_BIN = "python3"
FT_PATH = Path("/opt/FasterTransformer")
FT_BACKEND_PATH = Path("/opt/fastertransformer_backend")

# for debugging
FT_PATH_WITH_BUILD = FT_PATH
FT_PATH = Path(os.environ.get("FT_PATH", FT_PATH))


class Deployment(NemoMegatronStage):
    """
    FasterTransformer backend serving stage.
    """

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "deployment"
        self.stage_cfg = cfg.get("deployment")

    def make_stage_command_groups(self, stage_cfg_path, sub_stage=None,) -> List[List[str]]:
        """
        Make the command groups for current stage
        Command groups is a list of command group. A command group is defined as:
              0. Command group is a list of command strings
              1. Each command group occupies one bcprun, srun or bash
              2. Each command group eventually has multiple commands connected by ";"

        :param Path stage_cfg_path: path to interpolated and saved configuration
        :return: command groups for current stage
        :rtype: List[List[str]]
        """
        command_groups = [[]]

        command_groups[0] += self._make_sub_stage_command(sub_stage)
        command_groups = clean_command_groups(command_groups)
        return command_groups

    def _make_sub_stage_command(self, sub_stage):
        """
        Make the command group for current stage
        It occupies one bcprun, srun or bash.

        :return: command group for current stage
        :rtype: List[List[str]]
        """
        choice_model_type, choice_name = self.get_stage_config_choice()
        cmds_fn = {
            "deploy": {
                "gpt3": self._get_gpt_deployment_cmds,
                "t5": self._get_t5_deployment_cmds,
                "mt5": self._get_t5_deployment_cmds,
            },
        }[sub_stage][choice_model_type]
        return cmds_fn(self.cfg)

    def _make_sub_stages(self):
        sub_stages = ["convert"]
        return sub_stages

    def setup_folder_and_data(self) -> None:
        """Setup job/data folders and fine-tuning/prompt-learning dataset"""
        """Setup required folders and dataset"""
        job_path = self.get_job_path()
        job_path.folder.mkdir(parents=True, exist_ok=True)
        results_folder = job_path.results_folder
        results_folder.mkdir(parents=True, exist_ok=True)

    def run(self) -> str:
        """Execute export stage"""
        # Setup folders and datasets
        self.setup_folder_and_data()

        sub_stages = self._make_sub_stages()
        job_id = ""
        for sub_stage in sub_stages:
            # Save stage hydra config
            job_path = self.get_job_path(sub_stage)
            job_path.folder.mkdir(parents=True, exist_ok=True)

            stage_cfg_path = NemoMegatronStage.save_stage_hydra_config(self.stage_cfg, job_path)
            if job_id:
                dependency = f"aftercorr:{job_id}"
                self.stage_cfg["run"]["dependency"] = dependency

            # Make cluster parameters
            cluster_parameters = self._make_cluster_parameters(self.cluster, sub_stage)

            # Make command groups
            command_groups = self.make_stage_command_groups(stage_cfg_path, sub_stage)
            # Create launcher
            launcher = AutoLauncher(folder=job_path.folder, cluster=self.cluster, **cluster_parameters,)
            job_id = launcher.launch(command_groups=command_groups)

        return job_id

    def _make_cluster_parameters(self, cluster: str, sub_stage=None,) -> Dict:
        """Prepare cluster configuration"""
        cfg = self.cfg
        stage_cfg = self.stage_cfg

        run_cfg = stage_cfg.get("run")

        job_name = run_cfg.get("name")
        time_limit = run_cfg.get("time_limit")
        dependency = run_cfg.get("dependency")

        container_image = cfg.get("container")
        container_mounts = self._make_container_mounts_string()

        nodes = 1
        ntasks_per_node = 1

        setup = None
        env_vars = self.get_env_vars()
        if env_vars:
            setup = [f"export {k}={v}" for k, v in env_vars.items()]

        cluster_parameters = {}
        shared_parameters = {
            "job_name": job_name,
            "nodes": nodes,
            "time": time_limit,
            "ntasks_per_node": ntasks_per_node,
            "setup": setup,
        }
        if cluster == "bcm":
            cluster_cfg = cfg.get("cluster")
            slurm_cfg = {**copy.deepcopy(cluster_cfg)}
            job_name_prefix = slurm_cfg.pop("job_name_prefix")
            cluster_parameters = {**slurm_cfg}
            cluster_parameters.update(
                {
                    **shared_parameters,
                    "dependency": dependency,
                    "container_image": container_image,
                    "container_mounts": container_mounts,
                }
            )
            cluster_parameters["job_name"] = job_name_prefix + cluster_parameters["job_name"]
        elif cluster == "bcp":
            cluster_parameters.update(
                {**shared_parameters, "env_vars": env_vars,}
            )
        elif cluster == "interactive":
            cluster_parameters.update(shared_parameters)

        return cluster_parameters

    def _get_gpt_deployment_cmds(self, cfg):
        """ Generate deploy commands for GPT-3 models"""
        raise NotImplementedError

    def _get_t5_deployment_cmds(self, cfg):
        """ Generate deploy commands for T5/mT5 models"""
        raise NotImplementedError
