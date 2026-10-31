"""Provenance-aware ingestion and bounded context over Graphiti search results."""

import json
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

from graphiti_core.nodes import EpisodeType


class ProductFeature(BaseModel):
    """A product capability, including its intended and implemented behavior."""


class ProductSpec(BaseModel):
    """A specification or product requirement."""


class CodeArtifact(BaseModel):
    """A source file, symbol, migration, or repository artifact."""


class Screen(BaseModel):
    """A UI screen or user-facing state."""


class Mechanic(BaseModel):
    """A product rule, executable contract, or behavior."""


class RuntimeObservation(BaseModel):
    """An observed execution, test result, or telemetry event."""


class Decision(BaseModel):
    """A decision and its rationale or status."""


class AgentEpisode(BaseModel):
    """An agent action, investigation, or outcome."""


ENTITY_TYPES = {
    cls.__name__: cls
    for cls in (
        ProductFeature,
        ProductSpec,
        CodeArtifact,
        Screen,
        Mechanic,
        RuntimeObservation,
        Decision,
        AgentEpisode,
    )
}


class SourceKind(str, Enum):
    SPEC = 'spec'
    CODE = 'code'
    SCREEN = 'screen'
    CONTRACT = 'contract'
    OBSERVATION = 'observation'
    DECISION = 'decision'
    AGENT_EPISODE = 'agent_episode'


@dataclass(frozen=True)
class Evidence:
    kind: SourceKind
    uri: str
    revision: str
    content: str
    observed_at: datetime
    locator: str = ''

    def __post_init__(self):
        if not all((self.uri.strip(), self.revision.strip(), self.content.strip())):
            raise ValueError('uri, revision, and content are required')
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError('observed_at must have a timezone')


async def ingest(client: Any, evidence: Evidence, group_id: str):
    """Write an evidence envelope as a Graphiti episode; extraction is probabilistic."""
    if not group_id.strip():
        raise ValueError('group_id is required')
    data = asdict(evidence)
    data['kind'] = evidence.kind.value
    data['observed_at'] = evidence.observed_at.isoformat()
    return await client.add_episode(
        name=f'{evidence.kind.value}: {evidence.uri}@{evidence.revision}',
        episode_body=json.dumps(data),
        source_description=f'{evidence.uri}@{evidence.revision} {evidence.locator}',
        reference_time=evidence.observed_at,
        source=EpisodeType.json,
        group_id=group_id,
        entity_types=ENTITY_TYPES,
        custom_extraction_instructions=(
            'Distinguish intended, implemented, and observed behavior. Preserve uncertainty; '
            'do not infer that a planned feature is implemented.'
        ),
    )


def traverse(edges: list[Any], start_uuid: str, max_hops: int = 2) -> list[Any]:
    """Return edges reachable in an already retrieved subgraph, in breadth-first order."""
    if not 0 <= max_hops <= 5:
        raise ValueError('max_hops must be between 0 and 5')
    adjacency: dict[str, list[tuple[int, str]]] = {}
    for i, edge in enumerate(edges):
        adjacency.setdefault(edge.source_node_uuid, []).append((i, edge.target_node_uuid))
        adjacency.setdefault(edge.target_node_uuid, []).append((i, edge.source_node_uuid))
    seen_nodes = {start_uuid}
    seen_edges: set[int] = set()
    queue = deque([(start_uuid, 0)])
    result = []
    while queue:
        node, depth = queue.popleft()
        if depth == max_hops:
            continue
        for index, other in adjacency.get(node, []):
            if index not in seen_edges:
                seen_edges.add(index)
                result.append(edges[index])
            if other not in seen_nodes:
                seen_nodes.add(other)
                queue.append((other, depth + 1))
    return result


async def context(
    client: Any,
    query: str,
    group_id: str,
    center_node_uuid: str | None = None,
    max_facts: int = 20,
    max_hops: int = 2,
) -> list[Any]:
    """Search a single partition, optionally restricting the returned subgraph by reachability."""
    if not group_id.strip() or not query.strip() or not 1 <= max_facts <= 100:
        raise ValueError('query, group_id and max_facts (1..100) are required')
    edges = await client.search(
        query=query, group_ids=[group_id], num_results=max_facts, center_node_uuid=center_node_uuid
    )
    return traverse(edges, center_node_uuid, max_hops) if center_node_uuid else edges
