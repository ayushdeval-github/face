# 1. Use the Python version you specified in your README
FROM python:3.12-slim

# 2. Hugging Face Spaces run as a non-root user. 
# Creating this user is a best practice to avoid permission errors.
RUN useradd -m -u 1000 user
USER user

# 3. Set the home directory and path
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 4. Set the working directory inside the container
WORKDIR $HOME/app

# 5. Copy your requirements file first
# (This helps Docker cache the installed packages so future builds are much faster)
COPY --chown=user requirements.txt .

# 6. Install the dependencies listed in your requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of your project files (templates, utils, models, etc.)
COPY --chown=user . .

# 8. Expose the required Hugging Face port
EXPOSE 5000

# 9. Start your Flask application
CMD ["python", "app.py"]