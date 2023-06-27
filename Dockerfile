FROM 045411470840.dkr.ecr.us-east-1.amazonaws.com/stable-diffusion-server:base

COPY ./server ./server

CMD ["python3", "server"]