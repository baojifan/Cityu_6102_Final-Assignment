from neo4j import GraphDatabase

# Connect to the Neo4j database
uri = "neo4j://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "1263574116Ab"))


def load_data():
    with driver.session() as session:
        # Clear the database, deleting all nodes and relationships
        session.run("MATCH (n) DETACH DELETE n")

        # Load user data, create User nodes with attributes: id, gender, age, occupation, zip_code
        print("Loading user data...")
        with open("users.dat", 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                tokens = line.split('::')
                if len(tokens) != 5:
                    continue
                user_id, gender, age, occupation, zip_code = tokens
                session.run(
                    """
                    CREATE (:User {id: $user_id, gender: $gender, age: $age, occupation: $occupation, zip_code: $zip_code})
                    """,
                    user_id=int(user_id),
                    gender=gender,
                    age=int(age),
                    occupation=occupation,
                    zip_code=zip_code
                )

        # Load movie data, create Movie nodes with id and title attributes, and create relationships with Genre nodes
        print("Loading movie data...")
        with open("movies.dat", 'r', encoding='ISO-8859-1') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                tokens = line.split('::')
                if len(tokens) != 3:
                    continue
                movie_id, title, genres = tokens
                session.run(
                    """
                    CREATE (:Movie {id: $movie_id, title: $title})
                    """,
                    movie_id=int(movie_id),
                    title=title
                )
                genre_list = genres.split('|')
                for genre in genre_list:
                    session.run(
                        """
                        MERGE (g:Genre {name: $genre})
                        WITH g
                        MATCH (m:Movie {id: $movie_id})
                        CREATE (m)-[:HAS_GENRE]->(g)
                        """,
                        genre=genre,
                        movie_id=int(movie_id)
                    )

        # Load rating data, create RATED relationships between User and Movie nodes with rating and timestamp attributes
        print("Loading rating data...")
        with open("ratings.dat", 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                tokens = line.split('::')
                if len(tokens) != 4:
                    continue
                user_id, movie_id, rating, timestamp = tokens
                session.run(
                    """
                    MATCH (u:User {id: $user_id})
                    MATCH (m:Movie {id: $movie_id})
                    CREATE (u)-[:RATED {rating: $rating, timestamp: $timestamp}]->(m)
                    """,
                    user_id=int(user_id),
                    movie_id=int(movie_id),
                    rating=int(rating),
                    timestamp=int(timestamp)
                )


if __name__ == "__main__":
    load_data()
    print("Data loading completed.")
    print("hello world")