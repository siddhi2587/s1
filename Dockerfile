FROM alpine:latest
WORKDIR /app
COPY . .
CMD ["echo", "Docker build successful!"]
