# EnerVision AI - Frontend Production Dockerfile
# Stage 1: Build React app with Node.js
# Stage 2: Serve static files with Nginx

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files first for better layer caching
COPY package.json package-lock.json ./

RUN npm ci --frozen-lockfile

# Copy source code
COPY . .

# Build production bundle
RUN npm run build

# ── Stage 2: Production Nginx server ─────────────────────────────────────────
FROM nginx:1.25-alpine AS production

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom nginx config for SPA routing
COPY nginx/spa.conf /etc/nginx/conf.d/default.conf

# Copy built React app from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Create non-root user
RUN addgroup -g 1001 -S nginx_app && \
    adduser -S -u 1001 -G nginx_app nginx_app

# Set proper permissions
RUN chown -R nginx_app:nginx_app /usr/share/nginx/html && \
    chown -R nginx_app:nginx_app /var/cache/nginx && \
    chown -R nginx_app:nginx_app /var/log/nginx && \
    touch /var/run/nginx.pid && \
    chown -R nginx_app:nginx_app /var/run/nginx.pid

USER nginx_app

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD wget -qO- http://127.0.0.1:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
