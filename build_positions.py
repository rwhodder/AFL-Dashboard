"""One-off script to pre-fill player_positions.csv with classified positions."""
import pandas as pd

df = pd.read_csv('player_positions.csv')

GENF_MAP = {
    # ADE
    'Ben Keays':           ('FwdMid', ''),
    'Darcy Fogarty':       ('KeyF',   'Tall forward - reclassify to KeyF'),
    'Izak Rankine':        ('SmF',    ''),
    'Luke Pedlar':         ('SmF',    ''),
    'Riley Thilthorpe':    ('KeyF',   'Tall forward - reclassify to KeyF'),
    # BRL
    'Charlie Cameron':     ('SmF',    ''),
    'Kai Lohmann':         ('MedF',   ''),
    'Lincoln McCarthy':    ('MedF',   ''),
    'Ty Gallop':           ('SmF',    ''),
    'Zac Bailey':          ('SmF',    ''),
    # CAR
    'Ashton Moir':         ('MedF',   ''),
    'Ben Ainsworth':       ('SmF',    ''),
    'Brodie Kemp':         ('MedF',   ''),
    'Harry McKay':         ('KeyF',   'Tall forward - reclassify to KeyF'),
    'Will Hayward':        ('MedF',   ''),
    # COL
    'Beau McCreery':       ('SmF',    ''),
    'Daniel McStay':       ('MedF',   ''),
    'Lachie Schultz':      ('SmF',    ''),
    'Roan Steele':         ('MedF',   ''),
    # ESS
    'Hussien El Achkar':   ('MedF',   'REVIEW'),
    'Isaac Kako':          ('SmF',    ''),
    'Jayden Nguyen':       ('SmF',    ''),
    'Kyle Langford':       ('MedF',   ''),
    'Sam Durham':          ('FwdMid', ''),
    # FRE
    'Isaiah Dudley':       ('MedF',   ''),
    'Jye Amiss':           ('KeyF',   'Tall forward developing as KeyF'),
    'Murphy Reid':         ('SmF',    ''),
    'Sam Switkowski':      ('SmF',    ''),
    # GCS
    'Bailey Humphrey':     ('SmF',    ''),
    'Ethan Read':          ('MedF',   ''),
    'Leonardo Lombard':    ('MedF',   ''),
    'Touk Miller':         ('FwdMid', ''),
    'Will Graham':         ('FwdMid', ''),
    # GEE
    'Brad Close':          ('MedF',   ''),
    'Jack Martin':         ('FwdMid', ''),
    'Jeremy Cameron':      ('KeyF',   'Tall forward - reclassify to KeyF'),
    'Mitch Knevitt':       ('MedF',   ''),
    'Patrick Dangerfield': ('FwdMid', 'Plays forward-mid split'),
    # GWS
    'Callum Brown':        ('SmF',    ''),
    'Jake Riccardi':       ('MedF',   ''),
    'Jake Stringer':       ('MedF',   ''),
    'Josaia Delana':       ('FwdMid', ''),
    'Max Gruzewski':       ('MedF',   'REVIEW'),
    'Stephen Coniglio':    ('FwdMid', 'Moved forward but primarily a mid'),
    "Xavier O'Halloran":   ('FwdMid', ''),
    # HAW
    'Jack Ginnivan':       ('SmF',    ''),
    'Jack Gunston':        ('MedF',   ''),
    'Mitch Lewis':         ('KeyF',   'Tall forward - reclassify to KeyF'),
    'Nick Watson':         ('SmF',    ''),
    # MEL
    'Bayley Fritsch':      ('MedF',   ''),
    'Ed Langdon':          ('SmF',    ''),
    'Koltyn Tholstrup':    ('MedF',   'REVIEW'),
    'Kysaiah Pickett':     ('SmF',    ''),
    # NTH
    'Cameron Zurhaar':     ('MedF',   ''),
    'Jacob Konstanty':     ('SmF',    ''),
    'Lachy Dovaston':      ('SmF',    ''),
    'Luke Parker':         ('FwdMid', 'Plays forward-mid split'),
    'Paul Curtis':         ('FwdMid', ''),
    # PTA
    'Corey Durdin':        ('MedF',   ''),
    'Darcy Byrne-Jones':   ('GenD',   'Defender - reclassify to GenD'),
    'Jack Lukosius':       ('FwdMid', ''),
    'Jason Horne-Francis': ('FwdMid', 'Plays forward-mid split'),
    'Joe Berry':           ('SmF',    ''),
    'Joe Richards':        ('MedF',   ''),
    # RIC
    'Harry Armstrong':     ('SmF',    ''),
    'Jonty Faull':         ('MedF',   ''),
    'Maurice Rioli':       ('SmF',    ''),
    'Rhyan Mansell':       ('SmF',    ''),
    'Sam Lalor':           ('MedF',   ''),
    'Seth Campbell':       ('SmF',    ''),
    # STK
    'Hugo Garcia':         ('SmF',    ''),
    'Jack Higgins':        ('SmF',    ''),
    'Liam Ryan':           ('SmF',    ''),
    'Mattaes Phillipou':   ('FwdMid', ''),
    'Mitch Owens':         ('MedF',   ''),
    'Rowan Marshall':      ('Ruck',   'Ruck-forward - reclassify to Ruck'),
    # SYD
    'Isaac Heeney':        ('FwdMid', ''),
    'Jake Lloyd':          ('FwdMid', ''),
    'Joel Amartey':        ('MedF',   ''),
    'Malcolm Rosas':       ('SmF',    ''),
    'Tom Papley':          ('SmF',    ''),
    # WBD
    'Arthur Jones':        ('MedF',   ''),
    'Cooper Hynes':        ('SmF',    'REVIEW'),
    'Joel Freijah':        ('SmF',    ''),
    'Jordan Croft':        ('MedF',   'REVIEW'),
    'Oskar Baker':         ('MedF',   ''),
    'Rhylee West':         ('MedF',   ''),
    'Tom Liberatore':      ('FwdMid', ''),
    # WCE
    'Archer Reid':         ('SmF',    ''),
    'Jack Graham':         ('FwdMid', ''),
    'Jamie Cripps':        ('SmF',    ''),
    'Willem Duursma':      ('SmF',    ''),
}

for idx, row in df.iterrows():
    if row['position'] == 'GenF' and row['player'] in GENF_MAP:
        new_pos, note = GENF_MAP[row['player']]
        df.at[idx, 'position'] = new_pos
        df.at[idx, 'notes'] = note

# Mark any remaining GenF for review
still_genf = df['position'] == 'GenF'
df.loc[still_genf, 'notes'] = 'REVIEW: classify as SmF / MedF / FwdMid'

df.to_csv('player_positions.csv', index=False)

print('Position counts after pre-fill:')
print(df['position'].value_counts().to_string())
print()
print('Players still needing your review:')
review = df[df['notes'].str.contains('REVIEW', na=False)]
print(review[['player', 'team', 'position', 'notes']].to_string(index=False))
