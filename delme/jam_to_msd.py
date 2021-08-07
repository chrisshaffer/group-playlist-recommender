import whoosh
import whoosh.fields
import whoosh.index
import whoosh.analysis
import whoosh.qparser

# Path to a whoosh index of the Million Song Dataset.
# You'll need to create one of these if you don't have one already, as an
# example see
# https://github.com/craffel/midi-dataset/blob/master/scripts/create_whoosh_indices.py
MSD_INDEX = '/home/craffel/projects/midi-dataset-cqt/data/msd/index/'
# Path to the jams.tsv file, get it from
# https://archive.org/details/thisismyjam-datadump
JAMS_TSV_PATH = 'archive/jams.tsv'
# Path to the output match TSV file
OUTPUT_TSV = 'jam_to_msd.tsv'


def search(searcher, schema, artist, title, threshold=26):
    ''' Find matches with a score above a certain threshold in a whoosh index.
    '''
    # Convert arguments to unicode
    if type(artist) != unicode:
        artist = unicode(artist, encoding='utf-8')
    if type(title) != unicode:
        title = unicode(title, encoding='utf-8')
    # Construct a query parser for the whoosh index
    arparser = whoosh.qparser.QueryParser('artist', schema)
    tiparser = whoosh.qparser.QueryParser('title', schema)
    q = whoosh.query.And([arparser.parse(artist), tiparser.parse(title)])
    # Get whoosh results
    results = searcher.search(q)

    if len(results) > 0:
        # If there were any results, return the ones with a score above the
        # provided threshold
        return [[r['track_id'], r['artist'], r['title']] for r in results if
                r.score > threshold]
    else:
        # If there were no results, return an empty list
        return []

if __name__ == '__main__':

    # Load in the jams tsv file
    with open(JAMS_TSV_PATH) as f:
        # Parse each line in the file
        jams = [line.strip().split('\t') for line in f]
    # Remove header row
    jams = jams[1:]

    # Load in the whoosh index
    msd_index = whoosh.index.open_dir(MSD_INDEX)

    # Open the output tsv file for writing
    with open(OUTPUT_TSV, 'wb') as jam_to_msd_tsv:
        # Open a searcher object from the index
        with msd_index.searcher() as searcher:
            # Match each jam entry
            for jam in jams:
                try:
                    # Extract artist and title from the jam entry
                    artist, title = jam[2:4]
                    # Match this artist and title against the MSD
                    results = search(searcher, msd_index.schema, artist, title)
                    # Write out each result to the output TSV file
                    for result in results:
                        jam_to_msd_tsv.write(
                            '{}\t{}\n'.format(jam[0], result[0]))
                        print u"{}: {} - {} -> {} - {}".format(
                            result[0], result[1], result[2], artist, title)
                except Exception as e:
                    print "Error: {}, jam={}".format(e, jam)