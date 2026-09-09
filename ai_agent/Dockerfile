FROM node:20-alpine

WORKDIR /usr/src/app

# Copy package files
COPY package*.json ./

# Install all dependencies (we need tsx from devDependencies to run)
RUN npm ci

# Copy the rest of the application
COPY . .

# Expose port (Vertex AI default is 8080)
EXPOSE 8080

# Run the server using tsx
CMD ["npx", "tsx", "server.ts"]
