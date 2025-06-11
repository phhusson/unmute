# STT & TTS project page

The project page for text-to-speech and speech-to-text, with demos.
This is a Next.js project.

Install git hooks using:
```bash
pre-commit install
pre-commit install --hook-type pre-push
```

Use `pnpm` to install:

```bash
pnpm install
# if you don't have Node:
pnpm env use --global lts
```

Then run:

```bash
pnpm run dev
```

For deployment:

```bash
docker build -t stt-tts-project-page .
```

You'll also need to build the backend Docker image, see the root of this repo.

Then run the whole system using Docker Compose:

```bash
docker compose up
```

## Self-signed HTTPS for development

When deploying a dev version, we need HTTPS for the browser to allow microphone access even on non-localhost domains.
We use self-signed certificates which doesn't add any security but fixes the microphone issue.

Run:

```bash
mkdir certs
openssl req -x509 -newkey rsa:4096 -keyout certs/selfsigned.key -out certs/selfsigned.crt -days 365 -nodes -subj "/CN=localhost"
```