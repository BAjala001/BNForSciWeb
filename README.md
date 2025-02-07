# BNForSci - Bayesian Network Software

A web-based Bayesian Network software for scientific applications, built with Django and modern web technologies.

## Features

- Create and edit Bayesian Networks through an intuitive web interface
- Calculate marginal probabilities and analyze network dependencies
- Interactive network visualization using D3.js
- Set findings and observe their effects on the network
- Modern, responsive user interface

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd bnforsci
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Apply database migrations:
```bash
python manage.py migrate
```

5. Create a superuser (optional):
```bash
python manage.py createsuperuser
```

## Running the Application

1. Start the development server:
```bash
python manage.py runserver
```

2. Open your web browser and navigate to:
```
http://localhost:8000
```

## Usage

1. Create a New Network:
   - Click "Create Network" in the navigation bar
   - Add nodes with their names, labels, and values
   - Define parent relationships between nodes
   - Set conditional probability tables (CPTs)
   - Save the network

2. Analyze Network:
   - View the network visualization
   - Calculate marginal probabilities
   - Set findings on nodes
   - Observe updated probabilities

## Development

The project structure follows Django best practices:

```
bnforsci/
├── bayesnet_app/          # Main Django app
│   ├── templates/         # HTML templates
│   ├── static/           # Static files (CSS, JS)
│   ├── views.py          # View functions
│   └── urls.py           # URL routing
├── bayesnet_web/         # Django project settings
├── static/               # Global static files
└── manage.py            # Django management script
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built using Django web framework
- Network visualization powered by D3.js
- Bayesian Network algorithms based on pgmpy 