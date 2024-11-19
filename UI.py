import tkinter as tk
import tkinter.messagebox
from neo4j import GraphDatabase
import pandas as pd


def Recommended():
    global e1
    global e2
    global e3
    global e4
    global t2

    window1 = tk.Tk()
    window1.title('Movie recommendations for the main user')
    window1.geometry('1000x800')

    title4 = tk.Label(window1, text='User-based movie recommendations',
                      font=('Arial', 30), width=40, height=2)
    title4.pack()
    title5 = tk.Label(window1, text='Please enter the user',
                      font=('Arial', 20)).place(x=50, y=130, anchor='nw')
    title6 = tk.Label(window1, text='User ID',
                      font=('Arial', 15)).place(x=50, y=180, anchor='nw')
    title7 = tk.Label(window1, text='Numbers of recommended movies: (1-20)',
                      font=('Arial', 15)).place(x=50, y=260, anchor='nw')
    title9 = tk.Label(window1, text='Need to filter? (1/0)',
                      font=('Arial', 15)).place(x=50, y=340, anchor='nw')
    title10 = tk.Label(window1, text='What type do you prefer?',
                       font=('Arial', 15)).place(x=50, y=420, anchor='nw')
    title11 = tk.Label(window1, text='Recommended movie list',
                       font=('Arial', 15)).place(x=450, y=225, anchor='nw')

    # Input fields
    e1 = tk.Entry(window1, show=None, font=('Arial', 14))
    e1.place(x=90, y=220, anchor='nw')
    e2 = tk.Entry(window1, show=None, font=('Arial', 14))
    e2.place(x=90, y=300, anchor='nw')
    e3 = tk.Entry(window1, show=None, font=('Arial', 14))
    e3.place(x=90, y=380, anchor='nw')
    e4 = tk.Entry(window1, show=None, font=('Arial', 14))
    e4.place(x=90, y=460, anchor='nw')

    b5 = tk.Button(window1, text='Click to recommend', font=('Arial', 12),
                   width=20, height=2, command=queries)
    b5.place(x=105, y=500, anchor='nw')

    t2 = tk.Text(window1, width=150, height=30)
    t2.place(x=350, y=250, anchor='nw')

    window1.mainloop()


def queries():
    t2.delete(1.0, tk.END)
    t2.update()
    userid = e1.get()
    m = int(e2.get())
    fg = int(e3.get())

    while True:
        if m > 20:
            tk.messagebox.showinfo(title='Alarm', message='Exceeded the recommended number of movies!')
            break
        genres = []

        if int(fg):
            with driver.session() as session:
                try:
                    q = session.run("MATCH (g:Genre) RETURN g.name AS genre")
                    result = []
                    for i, r in enumerate(q):
                        result.append([i, r["genre"]])
                    df = pd.DataFrame(result, columns=["index", "genre"])

                    t2.insert('end', "Available film genres and their indices:\n")
                    t2.insert('end', df.to_string(index=False))
                    t2.insert('end', "\nPlease enter the type index (separated by a space) in ‘Type to filter’:\n")

                    inp = e4.get()
                    if len(inp.strip()) != 0:
                        inp = inp.strip().split(" ")
                        genres = []
                        invalid_indices = False
                        for x in inp:
                            try:
                                idx = int(x)
                                genre = df.loc[df['index'] == idx, 'genre'].values[0]
                                genres.append(genre)
                            except (ValueError, IndexError):
                                tk.messagebox.showinfo(title='Error', message=f'Invalid type index: {x}')
                                invalid_indices = True
                                break
                        if invalid_indices:
                            break
                    else:
                        tk.messagebox.showinfo(title='Notice', message='The filter information is empty!')
                        break
                except Exception as e:
                    t2.insert('end', f"Error: {e}\n")
                    break

        with driver.session() as session:  # Get user rating records
            q = session.run(f"""
                MATCH (u1:User {{id : {userid}}})-[r:RATED]-(m:Movie)
                RETURN m.title AS title, r.rating AS grade
                ORDER BY grade DESC
                """)
            result = []
            for r in q:
                result.append([r["title"], r["grade"]])

            if len(result) == 0:
                p2 = "The user's rating record was not found."
                t2.insert('end', f"{p2}\n")
            else:
                p1 = "The user's rating is recorded as follows:"
                t2.insert('end', f"{p1}\n")
                df_user = pd.DataFrame(result, columns=["title", "grade"])
                p3 = df_user.to_string(index=False)
                t2.insert('end', f"{p3}\n")

            # Delete previous similarity relations
            session.run(f"""
                MATCH (u1:User)-[s:SIMILARITY]-(u2:User)
                DELETE s
                """)

            # Calculate user similarity
            session.run(f"""
                MATCH (u1:User {{id : {userid}}})-[r1:RATED]-(m:Movie)
                      -[r2:RATED]-(u2:User)
                WITH
                    u1, u2,
                    COUNT(m) AS movies_common,
                    SUM(r1.rating * r2.rating) /
                    (SQRT(SUM(r1.rating^2)) * SQRT(SUM(r2.rating^2))) AS sim
                WHERE movies_common >= {movies_common} AND sim > {threshold_sim}
                MERGE (u1)-[s:SIMILARITY]-(u2)
                SET s.sim = sim
                """)

            Q_GENRE = ""
            if len(genres) > 0:
                Q_GENRE = "AND ((SIZE(gen) > 0) AND "
                genres_cypher_list = "[" + ", ".join(f"'{g}'" for g in genres) + "]"
                Q_GENRE += f"(ANY(x IN {genres_cypher_list} WHERE x IN gen)))"

            # Get recommended movies
            q = session.run(f"""
                MATCH (u1:User {{id : {userid}}})-[s:SIMILARITY]-(u2:User)
                WITH u1, u2, s
                ORDER BY s.sim DESC LIMIT {k}
                MATCH (m:Movie)-[r:RATED]-(u2)
                OPTIONAL MATCH (g:Genre)--(m)
                WITH u1, u2, s, m, r, COLLECT(DISTINCT g.name) AS gen
                WHERE NOT((m)-[:RATED]-(u1)) {Q_GENRE}
                      AND s.sim IS NOT NULL AND r.rating IS NOT NULL
                WITH
                    m.title AS title,
                    SUM(r.rating * s.sim)/SUM(s.sim) AS grade,
                    COUNT(u2) AS num,
                    gen
                WHERE num >= {users_common}
                RETURN title, grade, num, gen
                ORDER BY grade DESC, num DESC
                LIMIT {m}
                """)

            p4 = "\nThe recommended movie is as follows:"
            t2.insert('end', f"{p4}\n")

            result = []
            for r in q:
                # Limit genres list to a maximum of three types
                genres_limited = r["gen"][:3] if r["gen"] else []
                result.append([r["title"], r["grade"],
                               r["num"], genres_limited])
            if len(result) == 0:
                p6 = "No suitable recommendation found."
                t2.insert('end', f"{p6}\n")
                break
            df_result = pd.DataFrame(result,
                                     columns=["title", "avg grade",
                                              "num recommenders", "genres"])
            p5 = df_result.to_string(index=False)
            t2.insert('end', f"{p5}\n")
            break

# Connect to the Neo4j database
uri = "neo4j://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "1263574116Ab"))

# Set collaborative filtering parameters
k = 10             # Number of nearest users
movies_common = 2  # Minimum number of common rated movies
users_common = 1   # Minimum number of similar users rating a movie
threshold_sim = 0.9  # User similarity threshold

# Create main window
window = tk.Tk()
window.title('Main interface of the movie recommendation system')
window.geometry('800x600')

# Set labels
title = tk.Label(window, text='Movie recommendation system', font=('Arial', 40),
                 width=40, height=2)
title.pack()

# Place subtitles
title2 = tk.Label(window, text='Recommendation:', font=('Arial', 15))
title2.place(x=50, y=130, anchor='nw')
title3 = tk.Label(window, text='System operation information:', font=('Arial', 15))
title3.place(x=450, y=265, anchor='nw')

# Place buttons
b3 = tk.Button(window, text='Recommended by the main user', font=('Arial', 12),
               width=12, height=2, command=Recommended)
b3.place(x=80, y=200, anchor='nw')

# Place output box
t1 = tk.Text(window, width=40, height=15)
t1.place(x=450, y=315, anchor='nw')

# Main window loop
window.mainloop()