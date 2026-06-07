FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir ".[all]"

ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8000

EXPOSE 8000

CMD ["ohwise-mcp", "--transport", "sse"]
