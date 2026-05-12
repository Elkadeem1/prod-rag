import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const errorRate = new Rate('errors');
const chatLatency = new Trend('chat_latency', true);
const cragVerdicts = new Rate('crag_correct');

export const options = {
  scenarios: {
    smoke: { executor: 'constant-vus', vus: 1, duration: '30s', exec: 'smokeTest' },
    load: {
      executor: 'ramping-vus', startVUs: 0, exec: 'loadTest', startTime: '35s',
      stages: [
        { duration: '30s', target: 10 },
        { duration: '1m', target: 10 },
        { duration: '30s', target: 20 },
        { duration: '1m', target: 20 },
        { duration: '30s', target: 0 },
      ],
    },
    filtered: {
      executor: 'constant-vus', vus: 5, duration: '1m', exec: 'filteredTest', startTime: '4m30s',
    },
  },
  thresholds: {
    http_req_duration: ['p(95)<6000'],
    http_req_failed: ['rate<0.05'],
    errors: ['rate<0.1'],
    chat_latency: ['p(90)<5000'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const QUERIES = [
  'What is the main contribution?',
  'Describe the methodology.',
  'What are the key results?',
  'How does this compare to prior work?',
  'What are the limitations?',
];
const CONTENT_TYPES = ['methodology', 'results', 'background', 'conclusion'];

export function smokeTest() {
  const health = http.get(`${BASE_URL}/health`);
  check(health, {
    'healthy': (r) => r.status === 200,
    'crag enabled': (r) => JSON.parse(r.body).crag_enabled === true,
  });
  sleep(1);
}

export function loadTest() {
  const query = QUERIES[Math.floor(Math.random() * QUERIES.length)];
  const res = http.post(`${BASE_URL}/api/v1/chat`,
    JSON.stringify({ query }), { headers: { 'Content-Type': 'application/json' } }
  );
  chatLatency.add(res.timings.duration);
  errorRate.add(res.status !== 200);
  if (res.status === 200) {
    const body = JSON.parse(res.body);
    cragVerdicts.add(body.verdict === 'CORRECT');
    check(res, { 'has answer': () => body.answer !== undefined });
  }
  sleep(0.5);
}

export function filteredTest() {
  const query = QUERIES[Math.floor(Math.random() * QUERIES.length)];
  const filter = CONTENT_TYPES[Math.floor(Math.random() * CONTENT_TYPES.length)];
  const res = http.post(`${BASE_URL}/api/v1/chat`,
    JSON.stringify({ query, content_type_filter: filter }),
    { headers: { 'Content-Type': 'application/json' } }
  );
  chatLatency.add(res.timings.duration);
  errorRate.add(res.status !== 200);
  check(res, { 'filtered ok': (r) => r.status === 200 || r.status === 503 });
  sleep(0.5);
}