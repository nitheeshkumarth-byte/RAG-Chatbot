# Deploying to AWS (EC2, via the Console)

This deploys your existing project as-is — same `api.py`, same `static/index.html`,
same everything — onto a single EC2 server you control from the AWS Console.
No Docker, no Terraform, no AWS CLI required.

## Architecture

```
Your browser  →  http://<EC2 public IP>:8000  →  EC2 instance running
                                                    uvicorn api:app
                                                          │
                                                          ▼
                                              index.pkl on the instance's
                                              own disk (EBS volume) —
                                              persists across stop/start,
                                              gone if you terminate it
```

One instance, one process, local disk for the index — deliberately the
simplest thing that works. Scaling/load-balancing is a "next steps" concern,
not a day-one one.

## Cost note

A `t3.small` (2 GB RAM) runs ~$0.02/hour (~$15/month if left running 24/7).
**Stop** the instance when you're not using it — stopped instances aren't
billed for compute, only for the small EBS storage cost (~$0.80/month for
8 GB). **Terminate** it if you're done for good; that deletes everything
including `index.pkl`.

---

## Step 1 — Launch the EC2 instance

1. Go to the [EC2 Console](https://console.aws.amazon.com/ec2/) → **Launch instance**.
2. **Name**: `rag-project`
3. **AMI**: Ubuntu Server 22.04 LTS (search "Ubuntu" in the AMI picker)
4. **Instance type**: `t3.small` (2 GB RAM — comfortable headroom for the embedding model; `t2.micro`/free-tier works too but is tight on memory)
5. **Key pair**: click **Create new key pair**, name it `rag-project-key`, download the `.pem` file and keep it somewhere safe. (You can skip this and use EC2 Instance Connect instead — see Step 3 — but having a key pair as a backup connection method is worth it.)
6. **Network settings** → click **Edit**:
   - Allow SSH traffic from **My IP** (not "Anywhere" — no reason to expose SSH to the whole internet)
   - Click **Add security group rule**: Type = `Custom TCP`, Port range = `8000`, Source = `Anywhere` (0.0.0.0/0) — this is what lets your browser reach the app
7. **Storage**: bump from the default 8 GB to **16 GB** (the Python ML dependencies, mainly `torch`, take a few GB)
8. Click **Launch instance**.

## Step 2 — Connect to it

Easiest path, entirely in the console, no local SSH client needed:

1. Select your instance in the EC2 list → click **Connect**
2. Choose the **EC2 Instance Connect** tab → **Connect**
3. A terminal opens right in your browser, already logged in as `ubuntu`

(If that ever fails, the `.pem` file from Step 1 works as a fallback:
`ssh -i rag-project-key.pem ubuntu@<public-ip>`)

## Step 3 — Get your project onto the instance

Pick whichever is easier for you:

**Option A — you have the project on GitHub already:**
```bash
git clone https://github.com/yourusername/rag_project.git
cd rag_project
```

**Option B — upload the zip from your own machine.** From a terminal on
*your* computer (not the browser terminal), using the `.pem` file from Step 1:
```bash
scp -i rag-project-key.pem rag_project.zip ubuntu@<public-ip>:~
```
Then back in the EC2 browser terminal:
```bash
sudo apt install -y unzip
unzip rag_project.zip -d rag_project
cd rag_project
```

## Step 4 — Install dependencies

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
This takes a few minutes — `torch` is the biggest download.

## Step 5 — Add your Gemini key

```bash
cp .env.example .env
nano .env
```
Paste in `GEMINI_API_KEY=AIzaSy...`, then `Ctrl+O`, `Enter`, `Ctrl+X` to save and exit.

## Step 6 — Run it as a background service

Running `uvicorn` directly dies the moment you close the browser terminal.
Instead, register it as a `systemd` service so it starts on boot and
restarts automatically if it crashes.

```bash
sudo nano /etc/systemd/system/rag.service
```
Paste this in (adjust the path if your folder isn't `/home/ubuntu/rag_project`):
```ini
[Unit]
Description=RAG Project
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/rag_project
ExecStart=/home/ubuntu/rag_project/venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```
Save and exit, then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable rag
sudo systemctl start rag
sudo systemctl status rag     # should show "active (running)"
```

## Step 7 — Test it

Find your instance's **Public IPv4 address** on the EC2 console (instance
details page), then open in your browser:

```
http://<public-ip>:8000
```

You should see the same upload panel and question box you built locally.

## Useful commands afterward

```bash
sudo systemctl restart rag      # after you update code or .env
sudo journalctl -u rag -f       # live logs, e.g. to see Gemini errors
```

To update the code later: pull/upload the new files, then `sudo systemctl restart rag`.

---

## Optional next steps

- **Elastic IP** — a stopped/restarted instance gets a *new* public IP by
  default. EC2 Console → Elastic IPs → Allocate → Associate with your
  instance, so the address stays fixed.
- **Port 80 instead of :8000** — install `nginx` as a reverse proxy in
  front of uvicorn, so people just hit `http://your-ip` with no port
  number. Worth doing once you're past the "does it work" stage.
- **HTTPS + a real domain** — point a domain at the Elastic IP, then use
  `certbot` (Let's Encrypt) with the nginx setup above for a free SSL cert.
- **S3 for the index instead of local disk** — if you ever move to
  multiple instances behind a load balancer, `index.pkl` on local disk
  won't be shared between them. Not needed for a single instance.

## Troubleshooting

- **Can't reach the site in the browser** → almost always the security
  group. Confirm port 8000 is open to your IP (or Anywhere) in Step 1.6.
- **`pip install` runs out of disk** → you likely kept the default 8 GB.
  Resize the EBS volume from the EC2 console (Volumes → Modify Volume →
  16 GB), then `sudo growpart /dev/xvda 1 && sudo resize2fs /dev/xvda1`
  on the instance.
- **Service shows "failed" in `systemctl status`** → run
  `sudo journalctl -u rag -n 50` to see the actual Python traceback; it's
  usually a missing `.env` or a typo in the systemd file's paths.
