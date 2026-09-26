# SPDX-License-Identifier: Apache-2.0
# Encode versioned control selections; device admission checks availability and exact doses.
import struct
from .config import aotx_hex
from .errors import aotx_bad
from .json_wire import aotx_fields, aotx_integer


def aotx_control_selection(value):
    aotx_fields(value, {'schema', 'kind', 'qualification_sha256', 'dose'},
        {'schema', 'kind', 'qualification_sha256', 'dose'}, 'control')
    if value['schema'] != 'aotx.control.selection.v1': raise aotx_bad('control.schema')
    if value['kind'] != 'residual_vector': raise aotx_bad('control.kind')
    digest = aotx_hex(value['qualification_sha256'], 32, 'control.qualification_sha256')
    dose = aotx_integer(value['dose'], -40000, 40000, 'control.dose')
    if not any(digest) or not dose: raise aotx_bad('control')
    return struct.pack('<IIiI', 1, 1, dose, 0) + digest
