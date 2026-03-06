# Stage 1: build the React app
FROM node:20-alpine AS build

WORKDIR /app

COPY app/ui/package.json app/ui/package-lock.json* ./
RUN npm install

COPY app/ui/ ./
RUN npm run build

# Stage 2: serve with nginx
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY docker/frontend/nginx.conf /etc/nginx/conf.d/default.conf
