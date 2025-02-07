from django.db import models
import pickle
import json

# Create your models here.

class BayesNetModel(models.Model):
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    network_data = models.BinaryField()  # Store pickled BayesNet object
    
    def save_network(self, network):
        """Save a BayesNet object"""
        self.network_data = pickle.dumps(network)
        self.save()
    
    def load_network(self):
        """Load the BayesNet object"""
        if self.network_data:
            return pickle.loads(self.network_data)
        return None
    
    def __str__(self):
        return self.name
