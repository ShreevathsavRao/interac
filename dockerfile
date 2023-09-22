# Use the official Python image as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the required Python packages
RUN pip install -r requirements.txt

# Copy the application files into the container
COPY app.py app.py

# Create a templates folder in the container
RUN mkdir templates

# Copy the contents of the templates folder into the container
COPY templates/ templates/

# Install NLTK and other dependencies
RUN pip install nltk

# Add a command to download the NLTK stopwords resource when building the image
RUN python -c "import nltk; nltk.download('stopwords')"

# Add a command to download the NLTK stopwords resource when building the image
RUN python -c "import nltk; nltk.download('all')"

# Expose a port for your application (change to the appropriate port if needed)
EXPOSE 8080

# Run the application when the container starts
CMD ["python", "app.py"]

