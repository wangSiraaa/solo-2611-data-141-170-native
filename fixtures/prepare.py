from pathlib import Path
import sqlite3,sys,shutil
p=Path(sys.argv[1] if len(sys.argv)>1 else 'work-input');p.mkdir(exist_ok=False)
source=Path(__file__).parent
with sqlite3.connect(p/'report.sqlite') as db:
    db.execute('PRAGMA foreign_keys=ON')
    db.executescript((source/'seed.sql').read_text())
for name in ('query.sql','plan.json','allowed_changes.json'):shutil.copy2(source/name,p/name)
print(p.resolve())
