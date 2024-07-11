import os
import sys
import folder_paths

custom_nodes_path = os.path.join(folder_paths.base_path, "custom_nodes")
llm_opt_prompt_path = os.path.join(custom_nodes_path, "llm-optimization-prompt")
sys.path.append(llm_opt_prompt_path)

from .llm import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']