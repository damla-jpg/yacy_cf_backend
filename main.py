'''
This is the main file of the project. It is the entry point of the project.
'''

# Custom imports
from birbs.server import app

if __name__ == '__main__':
    app.run(debug=True, port=3001)