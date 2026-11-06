# Separate AWS deployment preparation

This is a prepared deployment path, not a deployed service. It does not change
the current Render services. Create no resources until the region, cost,
authentication/rate-limit plan, and deployment approval are agreed.

## Architecture

```text
Browser
  -> AWS Amplify Hosting (React/Vite, HTTPS)
  -> Caddy HTTPS reverse proxy on EC2
  -> Flask/Gunicorn container
      -> private Ollama container (llama3.1)
      -> TensorFlow EfficientNet model file on persistent host storage
      -> SentenceTransformers cache + FAISS (reconstructed from uploaded source)
```

The backend and Ollama containers share only a private Docker network.
Only Caddy publishes ports 80/443. The application health route checks Flask,
not whether Ollama, TensorFlow, or the embedding model is ready.

## Measured local artifact sizes and sizing

Measured in the local development environment:

| Artifact | Size |
|---|---:|
| Ollama `llama3.1:latest` | 4.92 GB |
| EfficientNet `.keras` file | 32.0 MB |
| `all-MiniLM-L6-v2` snapshot | 87.3 MB |
| Bundled PDF examples | 26.4 MB |

These are disk sizes, not peak memory measurements. Ollama inference needs
additional RAM/VRAM for the model context and KV cache. TensorFlow and Python
also use runtime memory beyond the 32 MB model file. The recent real Ollama
load sample was sequential and does not establish multi-user capacity.

**Sizing recommendation for an initial single-instance deployment:**
`g6.2xlarge` in `us-east-1` (8 vCPU, 32 GiB host RAM, one NVIDIA L4 with 24 GiB
VRAM). The GPU can accelerate Ollama; TensorFlow classification can run on CPU.
Install NVIDIA Container Toolkit and use `compose.gpu.yaml`. The 16 GiB
`g6.xlarge` may be tight for the combined Ollama, TensorFlow and FAISS processes.
A CPU-only `m7i.2xlarge` (8 vCPU, 32 GiB) is a lower-cost fallback, but LLM
latency will likely be considerably worse. Benchmark both on a short-lived,
approved instance before treating either estimate as a service-level guarantee.

## Approximate monthly AWS costs

Illustrative on-demand estimate in `us-east-1`, using 730 hours/month; prices
vary by region, account, purchase option and date. Verify the final design in
the [AWS Pricing Calculator](https://calculator.aws/).

| Item | Estimate |
|---|---:|
| `g6.2xlarge` compute, $0.9776/hour | $713.65/month |
| 150 GB gp3 EBS, about $0.08/GB-month | $12.00/month |
| One public IPv4, $0.005/hour | $3.65/month |
| **EC2 baseline before network and logs** | **about $729/month** |
| Amplify hosting | Free within eligible monthly allowances; otherwise usage-based |

The GPU instance price is a third-party listing and must be reconfirmed in the
calculator/console at approval time. A CPU `m7i.2xlarge` listing at $0.4032/hour
would be about $294.34/month compute, or approximately $310/month with the same
150 GB disk and public IPv4. It has no GPU and will be slower for Ollama.

EC2 data transfer out has an aggregate first 100 GB/month free allowance for
eligible AWS usage; additional transfer is commonly $0.09/GB for this tier.
Amplify currently lists free allowances of 1,000 build minutes, 5 GB stored,
and 15 GB served per month for eligible accounts; listed overage examples are
$0.01/build minute, $0.023/GB-month stored, and $0.15/GB served. CloudWatch
cost depends on log volume, retention, and metrics. Keep logs concise, set a
14-day retention policy, and check the regional price page. DNS/domain
registration, taxes, any NAT Gateway, ECR storage, S3 artifact storage, and
unexpected traffic are not included. A NAT Gateway can add meaningful fixed
cost; this proposal uses a public-subnet EC2 instance with a restrictive
security group instead.

Newer AWS accounts may receive promotional credits, but a GPU instance is not
a free-tier-sized workload and can consume credits quickly. Verify eligibility
in the account billing console. Create AWS Budget alerts (for example at 50%,
80%, and forecast/actual 100% of an approved monthly cap); an alert does not
automatically stop an EC2 instance.

Pricing references checked for this estimate:

- [EC2 On-Demand pricing and data transfer](https://aws.amazon.com/ec2/pricing/on-demand/)
- [G6.2xlarge listing used for the GPU estimate](https://instances.vantage.sh/aws/ec2/g6.2xlarge)
- [M7i.2xlarge listing used for the CPU estimate](https://instances.vantage.sh/aws/ec2/m7i.2xlarge)
- [EBS pricing](https://aws.amazon.com/ebs/pricing/)
- [VPC/public IPv4 pricing](https://aws.amazon.com/vpc/pricing/)
- [Amplify Hosting pricing](https://aws.amazon.com/amplify/pricing/)
- [CloudWatch pricing](https://aws.amazon.com/cloudwatch/pricing/)
- [AWS Free Tier terms](https://aws.amazon.com/free/)
- [AWS Budgets pricing](https://aws.amazon.com/aws-cost-management/aws-budgets/pricing/)

## Security and reliability decisions

- Require a registered domain for `API_DOMAIN`; Caddy obtains and renews a
  public TLS certificate. Open only inbound 80/443. Do not expose ports 5000 or
  11434 and do not open SSH; use SSM Session Manager.
- Amplify uses its managed HTTPS domain/certificate. Set its
  `VITE_API_BASE_URL` to `https://<API_DOMAIN>` and set backend `CORS_ORIGINS`
  to the exact Amplify origin(s), not `*`.
- CORS is not authentication. Current Flask routes are anonymous and have no
  request rate limit. Do not treat HTTPS as access control. Before a public
  production launch, approve an authentication/rate-limiting approach (for
  example Cognito/JWT at an API gateway or a separately reviewed application
  change). This is a deployment blocker for an unrestricted public API because
  anonymous callers could consume GPU capacity.
- Use an EC2 instance role, never root access keys or credentials baked into
  an image. Scope the role to SSM Session Manager, a specific model-artifact
  S3 object (`s3:GetObject` only), and the CloudWatch log group. If using ECR,
  add read-only pull permission for the chosen repository. Scope operator IAM
  separately to the exact EC2, security group, instance profile, ECR, Amplify,
  and log-group resources needed. Restrict `iam:PassRole`.
- Keep Ollama on a private container network. The internal endpoint has no
  authentication; never publish it directly.
- Docker binds uploads and the Hugging Face model cache to the EC2 data disk.
  The active uploaded document source and a small marker survive container
  restarts; FAISS embeddings are rebuilt lazily on the next RAG question.
  This supports one active document and one instance, not shared or
  multi-instance storage. Back up the data disk and restrict access because
  uploads may contain private content.
- A cold first RAG request may download the embedding model. Keep the mounted
  Hugging Face cache, or pre-populate it from an approved S3 artifact. Health
  status alone does not prove model readiness.
- The Compose health check verifies only `/api/health`. Test an actual image
  request and RAG upload/query after starting the containers.
- `MAX_CONTENT_LENGTH` defaults to 32 MiB. Confirm the chosen reverse proxy and
  client timeouts permit model generation (up to 180 seconds in current RAG
  code). Gunicorn allows 210 seconds for a request.

## Prepare local files

1. Copy `deployment/aws/.env.aws.example` to the backend root as `.env.aws`.
   Set `API_DOMAIN`, `AWS_REGION`, and the exact Amplify origin. The file is
   ignored by Git. It contains configuration only; never put AWS access keys in
   it. Keep `OLLAMA_API_KEY` blank for the private Ollama container.
2. On the EC2 host, clone/copy this branch to `/opt/roadsense`, create
   `/opt/roadsense/models`, `/opt/roadsense/uploads`, and
   `/opt/roadsense/runtime/huggingface`; make the latter two writable by UID
   10001. Install Docker Engine/Compose and NVIDIA Container Toolkit for G6.
3. Store the ignored 32 MB classifier model in a private S3 bucket. From an
   SSM session, use the instance role to download the approved object to
   `/opt/roadsense/models/efficientnet_best_model.keras`. Verify its SHA-256
   before starting. Do not put the model in Git or the Docker image.
4. From the backend root, after architecture approval:

   ```powershell
   docker compose --env-file .env.aws -f deployment/aws/compose.yaml -f deployment/aws/compose.gpu.yaml config
   docker compose --env-file .env.aws -f deployment/aws/compose.yaml -f deployment/aws/compose.gpu.yaml up -d --build
   docker compose -f deployment/aws/compose.yaml exec ollama ollama pull llama3.1
   ```

   The build command is intentionally not run as part of this preparation.
   First-time model download requires outbound internet and persistent storage.
5. In Amplify, connect the frontend repository's `feature/rag-evaluation`
   branch after it has been pushed with approval. `amplify.yml` is prepared at
   the frontend repository root. Add the build-time variable
   `VITE_API_BASE_URL=https://<API_DOMAIN>`, select the desired region, and
   enable HTTPS/custom-domain DNS only after the backend is ready.
6. Install/configure CloudWatch Agent using
   `cloudwatch-agent.json`. Create `/roadsense-ai/backend` with 14-day
   retention. Attach the instance profile; do not use static keys. Configure
   the log group before starting the agent or grant only the narrow
   create-log-group/log-stream/put-events actions needed.

## AWS permissions to prepare

- **Deployment operator:** least-privilege permissions to create/describe/start/
  stop the chosen EC2 instance, security group and role association; pass only
  the named instance role; push/pull only the specified ECR repository if
  selected; configure the exact Amplify app/branch/environment variables;
  create the specified CloudWatch log group/retention; and configure one
  budget. Use an IAM Identity Center/federated role, not root credentials.
- **EC2 instance profile:** `AmazonSSMManagedInstanceCore`; read access only to
  the model object in the private S3 bucket; and scoped CloudWatch Agent
  permissions. Add ECR read-only only if deploying from ECR.
- **Networking:** outbound access for package/image/model pulls and Ollama
  model acquisition; inbound 80/443 only. Restrict the security-group egress
  further after deciding whether public HTTPS model downloads or private S3
  endpoints will be used.

## Deploy, verify, and roll back

Deployment waits for explicit architecture/cost/security approval. Then:

1. Confirm domain DNS points to the EC2 address and security-group rules are
   limited to 80/443; configure SSM and the instance profile.
2. Copy the selected source branch to the host, retrieve and checksum the
   classifier model, check Compose configuration, build, and start containers.
3. Pull `llama3.1` once. Verify container health and
   `https://<API_DOMAIN>/api/health`.
4. From the separate Amplify app, test prediction with a representative road
   image, upload a small permitted document, ask a RAG question, and inspect
   CloudWatch logs for request status, latency and token metrics. Never log
   prompt/answer/document contents.
5. Restart only the backend container in a planned check; verify the uploaded
   active document is re-indexed on the next RAG question. Ollama weights,
   uploads, and caches use persistent storage.
6. Confirm the existing Render frontend/backend still operate independently;
   do not edit their environment or service configuration.

For rollback, record the previous image/source version before updating.
Redeploy that image/tag and preserve the EBS data and Ollama model volume.
Do not delete the EC2 instance or volumes until backup and rollback approval.
When the AWS environment is no longer needed, stop/terminate resources only
with explicit approval and verify EBS volumes, public IPv4, ECR images, log
groups, Amplify artifacts and DNS records to avoid ongoing charges.

## Existing evaluation and local load evidence

The 10-question retrieval run recorded Hit@1/3/5 and MRR@5 of 1.0; this is a
small document-level retrieval sample, not answer correctness. The latest
prompt comparison used real local `llama3.1` for 20/20 generations. Baseline
fact coverage was 83.33%, average latency 4,654.7 ms and mean actual tokens
1,095.3. Candidate coverage was 90.00%, latency 3,931.2 ms and mean actual
tokens 1,117.2. Manual accuracy and grounding remain unreviewed.

The completed mocked local API run handled 25/25 requests at concurrency 5:
37.202 requests/second, 29.96 ms average, 47.0 ms p95, and 0% errors. The
completed real local Ollama run handled 5/5 requests at concurrency 1:
0.099 requests/second, 9,962.4 ms average, 16,828 ms p95, and 0% errors.
These are local measurements, not AWS or production capacity guarantees.

## Junior AI Developer interview explanation

“I separated the React frontend from a Flask API. The API uses a trained
EfficientNet model for image classification. For document questions, it
extracts and chunks a PDF or text file, embeds the chunks with
SentenceTransformers, searches them with FAISS, then sends the selected
context to Ollama. I evaluated retrieval separately from answer quality and
compared the unchanged production prompt with an experimental prompt on the
same questions and context. I added request latency and token monitoring,
tested API behavior with Flask’s test client and mocked AI services, and ran
small local load checks separately from real inference. The AWS copy is being
prepared as containers behind HTTPS, with private model service networking and
persistent local storage; I’m waiting for cost, security, and account approval
before provisioning anything.”
