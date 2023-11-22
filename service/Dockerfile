from hair-base:latest
WORKDIR "/app"
COPY . .
#RUN apt install libsqlite3-dev
#RUN mkdir /app/facefitting/build
#RUN cd /app/facefitting/build && cmake .. && make -j4
#RUN ln -s /app /p2a
CMD ["bash","/app/run_services.sh"]
