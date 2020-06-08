import numpy as np
def update_position(self, int_method, D0):
    # use explicit euler to update 
    dx = dt*self.U
    if D0 > 0: dx = dx + np.sqrt(2*D0*dt)*np.random.normal(size=2)
    self.X = self.X + dx
    self.X_total = self.X_total + dx

    # check if still in the periodic box
    self.check_in_box()

    # store positions
    self.history_X.append(self.X)
    self.history_X_total.append(self.X_total)
Swimmer().update_position()
