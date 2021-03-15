#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
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
#

from graphscope.framework.app import AppAssets
from graphscope.framework.app import not_compatible_for

__all__ = [
    "louvain",
]


@not_compatible_for("arrow_property", "dynamic_property")
def louvain(graph, tolerance=3, min_progress=100):
    """Compute best partition on the `graph`.

    Args:
        graph (:class:`Graph`): A projected simple graph.
        tolerance:
        min_progress:

    Returns:
        :class:`VertexDataContext`: A context with each vertex assigned with the shortest distance from the src.

    Examples:

    .. code:: python

        import graphscope as gs
        s = gs.session()
        g = s.load_from('The parameters for loading a graph...')
        pg = g.project_to_simple(v_label='vlabel', e_label='elabel', v_prop=None, e_prop='distance')
        r = gs.louvain(pg)
        s.close()

    """
    return AppAssets(algo="louvain")(graph, tolerance, min_progress)
