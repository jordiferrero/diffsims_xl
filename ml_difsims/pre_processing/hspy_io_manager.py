import hyperspy.api as hs
import os
from typing import Union

import dagster._check as check
from dagster.config import Field
from dagster.config.source import StringSource
from dagster.core.definitions.events import AssetKey, AssetMaterialization
from dagster.core.definitions.metadata import MetadataEntry, MetadataValue
from dagster.core.errors import DagsterInvariantViolationError
from dagster.core.execution.context.input import InputContext
from dagster.core.execution.context.output import OutputContext
from dagster.core.storage.io_manager import IOManager, io_manager
from dagster.core.storage.memoizable_io_manager import MemoizableIOManager
from dagster.utils import PICKLE_PROTOCOL, mkdir_p
from dagster.utils.backcompat import experimental


@io_manager(config_schema={"base_dir": Field(StringSource, is_required=False)})
def hyperspy_io_manager(init_context):
    base_dir = init_context.resource_config.get(
        "base_dir", init_context.instance.storage_directory()
    )

    return HyperSpyIOManager(base_dir=base_dir)


class HyperSpyIOManager(MemoizableIOManager):

    def __init__(self, base_dir=None):
        self.base_dir = check.opt_str_param(base_dir, "base_dir")
        self.write_mode = "wb"
        self.read_mode = "rb"

    def _get_path(self, context: Union[InputContext, OutputContext]) -> str:
        """Automatically construct filepath."""
        if context.has_asset_key:
            path = context.get_asset_identifier()
        else:
            path = context.get_identifier()

        return os.path.join(self.base_dir, *path)

    def has_output(self, context):
        filepath = self._get_path(context)

        return os.path.exists(filepath)

    def handle_output(self, context, obj):

        check.inst_param(context, "context", OutputContext)

        filepath = self._get_path(context)

        # Ensure path exists
        mkdir_p(os.path.dirname(filepath))
        obj.save(filepath, overwrite=True)

        context.add_output_metadata({"path": MetadataValue.path(os.path.abspath(filepath))})

    def load_input(self, context):
        """Unpickle the file and Load it to a data object."""
        check.inst_param(context, "context", InputContext)

        filepath = self._get_path(context) + ".hspy"
        context.add_input_metadata({"path": MetadataValue.path(os.path.abspath(filepath))})

        return hs.load(filepath)



