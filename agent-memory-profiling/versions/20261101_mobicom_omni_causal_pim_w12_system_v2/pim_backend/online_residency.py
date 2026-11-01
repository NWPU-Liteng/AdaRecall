"""Horizon-free admission for a capacity-certified immutable weight program."""
from __future__ import annotations

from collections import Counter


class OnlineResidency:
    def __init__(self, costs, placement, *, capacity_bytes, reserved_bytes=0):
        if (type(capacity_bytes) is not int or type(reserved_bytes) is not int
                or not 0 <= reserved_bytes <= capacity_bytes):
            raise ValueError('integer capacity and nonnegative reservations required')
        if not set(placement).issubset(costs):
            raise ValueError('placement requires actual physical weight costs')
        for identity, cost in costs.items():
            if not identity or type(cost['resident_bytes']) is not int or cost['resident_bytes'] <= 0:
                raise ValueError('positive physical weight allocation required')
            for route in ('host', 'pim', 'placement'):
                phases = cost['phase_costs'][route]
                if (any(type(phases[field]) is not int for field in ('min_cycles', 'max_cycles'))
                        or not 0 <= phases['min_cycles'] <= phases['max_cycles']):
                    raise ValueError('ordered nonnegative integer cost envelope required')
        for identity in placement:
            cost = costs[identity]
            if (placement[identity]['resident_bytes'] != cost['resident_bytes']
                    or cost['phase_costs']['host']['min_cycles']
                    <= cost['phase_costs']['pim']['max_cycles']):
                raise ValueError('admission pool must have robust positive per-call gain')
        if sum(costs[identity]['resident_bytes'] for identity in placement) + reserved_bytes > capacity_bytes:
            raise ValueError('all potentially admitted weights must fit without eviction')
        self.costs = costs
        self.pool = frozenset(placement)
        self.resident = set()
        self.calls = Counter()
        self.rent_calls = Counter()
        self.resident_bytes = 0

    def dispatch(self, identity):
        if identity not in self.costs:
            raise ValueError('current observed Linear requires a registered physical identity')
        cost = self.costs[identity]
        phases = cost['phase_costs']
        prior_calls = self.calls[identity]
        delta = phases['host']['max_cycles'] - phases['pim']['max_cycles']
        buy = (identity in self.pool and identity not in self.resident
               and (self.rent_calls[identity] + 1) * delta >= phases['placement']['max_cycles'])
        if buy:
            self.resident.add(identity)
            self.resident_bytes += cost['resident_bytes']
        route = 'pim' if identity in self.resident else 'host'
        if identity in self.pool and route == 'host':
            self.rent_calls[identity] += 1
        self.calls[identity] += 1
        result = {'parameter_identity': identity, 'prior_observed_calls': prior_calls,
            'route': route, 'placement_now': buy, 'resident_bytes_after': self.resident_bytes,
            'admission_opportunity_cycles': (prior_calls + 1) * delta if identity in self.pool else None}
        for field in ('min_cycles', 'max_cycles'):
            result['host_' + field] = phases['host'][field]
            result['hybrid_' + field] = phases[route][field]
            result['placement_' + field] = phases['placement'][field] if buy else 0
        result['host_external_bytes'] = cost['host_external_bytes']
        result['hybrid_external_bytes'] = cost[route + '_external_bytes'] + (cost['resident_bytes'] if buy else 0)
        return result

    def certificate(self):
        records = []
        for identity in sorted(self.pool):
            phases = self.costs[identity]['phase_costs']
            calls = self.calls[identity]
            delta = phases['host']['max_cycles'] - phases['pim']['max_cycles']
            buy = phases['placement']['max_cycles']
            rent = self.rent_calls[identity] * delta
            overhead = rent + (buy if identity in self.resident else 0)
            oracle = min(calls * delta, buy)
            if overhead > 2 * oracle:
                raise ValueError('integer admission cost certificate failed')
            records.append({'parameter_identity': identity, 'observed_calls': calls,
                'host_rental_calls': self.rent_calls[identity], 'admitted': identity in self.resident,
                'pessimistic_envelope_delta_cycles': delta, 'purchase_max_cycles': buy,
                'online_incremental_max_envelope_cycles': overhead,
                'retrospective_fixed_pool_oracle_incremental_max_envelope_cycles': oracle,
                'twice_oracle_bound_pass': True})
        return {'weights': records, 'resident_bytes': self.resident_bytes,
            'future_call_count_read_by_dispatch': False,
            'scope': 'two-times bound on incremental cost above always-resident PIM in a fixed per-primitive max-cycle surrogate; not a comparison to all-host, arbitrary eviction policies or actual hardware',
            'no_loss_against_all_host_guaranteed': False,
            'unknown_work_included': False, 'end_to_end_speedup': None}
