from gpu_scheduler import allocator

while True:
    request = input()
    try:
        request = tuple(request.split(' '))
        response = allocator(*request)
        if response:
            print(' '.join((str(x) for x in response)))
    except (TypeError, ValueError):
        print('Please try again')
