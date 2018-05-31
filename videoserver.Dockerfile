# Dockerfile for the web interface.
# The web interface is served by an aiohttp web server.

# Use python:3.6.5-alpine as the base image
FROM python:3.6.5-alpine

# Copy contents of adminweb into the container's /opt folder
COPY adminweb /opt
WORKDIR /opt

# Install all Python requirements for the web server
RUN pip install -r requirements.txt

# Expose port 8080 on the container
# This should be the port the web server serves from
EXPOSE 8080

# Start the web server as the default container command
CMD ["python", "app.py"]
