
Pkg.add("Mongoc")

# https://github.com/felipenoris/Mongoc.jl
# https://felipenoris.github.io/Mongoc.jl/stable/
Pkg.add("LibPQ")

using LibPQ;
conn = Connection("""host = wrds-pgdata.wharton.upenn.edu
                     port = port
                     user='username'
                     password='password'
                     sslmode = 'require'
                     dbname = wrds
                  """)

conn = LibPQ.Connection("dbname=postgres user=$DATABASE_USER"; throw_error=false)
result = execute(
            conn,
            "SELECT typname FROM pg_type WHERE oid = 16";
            throw_error=false,
        )

# https://juliapackages.com/p/awss3

# https://github.com/invenia/LibPQ.jl
# https://juliapackages.com/p/libpq

# https://discourse.julialang.org/t/accessing-postgresql-via-julia/7031/5

# s3
# https://juliapackages.com/p/awss3
