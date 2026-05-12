# Build Stage
FROM golang:1.26.3-alpine3.23 AS builder
WORKDIR /app

# Install gcc and libc for CGO (Required for SQLite)
RUN apk add --no-cache gcc musl-dev

COPY go.mod go.sum ./
RUN go mod download

COPY main.go ./
# Compile with CGO enabled
RUN CGO_ENABLED=1 GOOS=linux go build -o server main.go

# Runtime Stage
FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/server .

EXPOSE 3333
CMD ["./server"]