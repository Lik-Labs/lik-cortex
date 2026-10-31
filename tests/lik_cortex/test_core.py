import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from graphiti_core.lik_cortex.core import Evidence, SourceKind, context, ingest, traverse


def edge(a, b):
    return SimpleNamespace(source_node_uuid=a, target_node_uuid=b)


def test_reachability_is_bounded_and_handles_cycles():
    edges = [edge('a', 'b'), edge('b', 'c'), edge('c', 'a'), edge('c', 'd'), edge('x', 'y')]
    assert traverse(edges, 'a', 1) == [edges[0], edges[2]]
    assert traverse(edges, 'a', 2) == [edges[0], edges[2], edges[1], edges[3]]
    assert traverse(edges, 'a', 0) == []
    with pytest.raises(ValueError):
        traverse(edges, 'a', 6)


def test_ingest_carries_provenance_and_typed_ontology():
    client = SimpleNamespace(add_episode=AsyncMock(return_value='ok'))
    item = Evidence(
        SourceKind.CODE,
        'github://repo/file',
        'abc',
        'content',
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        'L1',
    )
    assert asyncio.run(ingest(client, item, 'lik')) == 'ok'
    kwargs = client.add_episode.await_args.kwargs
    assert 'github://repo/file' in kwargs['episode_body']
    assert kwargs['reference_time'] == item.observed_at
    assert 'Mechanic' in kwargs['entity_types']
    with pytest.raises(ValueError):
        Evidence(SourceKind.CODE, 'uri', 'rev', 'body', datetime(2026, 1, 1))


def test_context_scopes_search_and_filters_disconnected_edges():
    edges = [edge('a', 'b'), edge('x', 'y')]
    client = SimpleNamespace(search=AsyncMock(return_value=edges))
    assert asyncio.run(context(client, 'review', 'lik', 'a')) == edges[:1]
    assert client.search.await_args.kwargs['group_ids'] == ['lik']
    with pytest.raises(ValueError):
        asyncio.run(context(client, 'review', ''))
