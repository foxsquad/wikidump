import queue


DB_NAME = 'dump.sqlite3'


def process(data_queue, write_queue):
    import os.path as p

    from lxml import etree
    from wikidump import DEFAULT_NAMESPACE
    from wikidump.cleaner import strip

    ns_mapping = {'ns': DEFAULT_NAMESPACE}

    title_path = './{%(ns)s}title' % ns_mapping
    text_path = './{%(ns)s}revision/{%(ns)s}text' % ns_mapping
    format_path = './{%(ns)s}revision/{%(ns)s}format' % ns_mapping

    try:
        src = data_queue.get()
        while src is not None:
            try:
                elm = etree.fromstring(src)

                title = elm.find(title_path).text
                src = elm.find(text_path).text
                if src[-1] != '\n':
                    src += '\n'

                print('processing title %s' % title)
                stripped_text = strip(src)
                write_queue.put([title, stripped_text])
            except ValueError as e:
                print('  --> Error while processing title %s. '
                      'Error detail: %s' % (title, e))
                with open(p.join('error-articles',  '%s.txt' % title),
                          mode='w',
                          encoding='utf-8') as f:
                    f.write(src)
            finally:
                src = data_queue.get()

    except KeyboardInterrupt:
        pass
    # <-- terminated state
    # <-- process joined here


def write(db_name, write_queue, commit_frequency=1000):
    import sqlite3

    conn = sqlite3.connect(db_name)

    i = 0
    try:
        data = write_queue.get()
        while data is not None:
            conn.execute('insert into wiki_text (title, content)'
                         ' values (?, ?)', data)
            i += 1
            if i % commit_frequency == 0:
                conn.commit()
                i = 0
            data = write_queue.get()
    except KeyboardInterrupt:
        pass
    finally:
        conn.commit()
        conn.close()
        del conn
    # <-- terminated state
    # <-- process joined here


def prepare_db(db):
    import sqlite3

    conn = sqlite3.connect(db)
    with conn:
        conn.executescript('''\
create table if not exists wiki_text (
    id      integer not null primary key asc autoincrement,
    title   varchar(512),
    content blod
);

create index if not exists wiki_title_idx on wiki_text (
    title
);
''')
    conn.close()


def flush_queue(q):
    while True:
        try:
            q.get(False)
        except (queue.Empty, ):
            break


def main():
    import time
    from multiprocessing import Process, Queue

    from wikidump.iterator import iter_read

    prepare_db(DB_NAME)

    data_queue = Queue(100)
    write_queue = Queue(10)

    processors = []

    for idx in range(3):
        processor = Process(target=process, args=(data_queue, write_queue,),
                            name='processor-%d' % idx)
        processor.start()
        processors.append(processor)

    writer = Process(target=write, args=(DB_NAME, write_queue,),
                     name='writer')
    writer.start()

    i = iter_read()
    try:
        for idx, elm_str in enumerate(i):
            data_queue.put(elm_str)

            if idx >= 1_000_000:
                break
    except KeyboardInterrupt:
        print('\nUser terminated. Will clean up...\n')
    finally:
        del i

        print('Waiting for all processor processes to join...', end='')
        live_processes = [p for p in processors if p.is_alive()]
        while live_processes:
            try:
                data_queue.put(None, False)
            except queue.Full:
                pass
            finally:
                time.sleep(0.5)
                live_processes = [p for p in processors if p.is_alive()]

        for p in processors:
            p.join()
            time.sleep(0.1)
        print('    done.')

        print('Waiting for writer process to join...', end='')
        while writer.is_alive():
            try:
                write_queue.put(None, False)
            except queue.Full:
                time.sleep(0.5)
                pass
        print('    done.')

    # This is used to clean up remained write action on all queues,
    # release locking on pending thread(s).
    try:
        flush_queue(data_queue)
        data_queue.cancel_join_thread()
        data_queue.close()
        data_queue.join_thread()
    except BrokenPipeError:
        print('broken pipe error while trying to clean up data queue.')

    try:
        flush_queue(write_queue)
        write_queue.cancel_join_thread()
        write_queue.close()
        write_queue.join_thread()
    except BrokenPipeError:
        print('borken pipe error while trying to clean up write queue.')


if __name__ == '__main__':
    main()
