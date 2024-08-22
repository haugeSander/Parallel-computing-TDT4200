CC = gcc # Specify compiler to use
CFLAGS = -Wall -Wextra -std=c99 # Setting compiler flags to use based on C standard
TARGET = resize_image
SRCS = main.c bitmap.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
