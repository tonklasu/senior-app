# Use an official Node.js image as the base image
FROM node:19.5.0-slim

# Set the working directory in the image
WORKDIR /app

# Copy the package.json and package-lock.json files to the image
COPY package*.json ./

# Install the dependencies
RUN npm install

# Copy the rest of the files to the image
COPY . .

# Build the Next.js app
RUN npm run build

# Expose the port 3000 to the host
EXPOSE 3000

# Define the command to run when the image is started
CMD [ "npm", "start" ]
# RUN npm run start



