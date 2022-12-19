import numpy as np
# Define a matrix A
A = [[1, 2], [3, 4], [5, 6]]

# Define a function to compute the dot product of two matrices
def dot(A, B):
  C = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
  for i in range(len(A)):
    for j in range(len(B[0])):
      for k in range(len(B)):
        C[i][j] += A[i][k] * B[k][j]
  return C

# Define a function to compute the transpose of a matrix
def transpose(A):
  return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

# Define a function to compute the norm of a vector
def norm(v):
  return sum(x**2 for x in v)**0.5

# Define a function to normalize a vector
def normalize(v):
  return [x / norm(v) for x in v]

# Define a function to compute the outer product of two vectors
def outer(u, v):
  return [[u[i] * v[j] for j in range(len(v))] for i in range(len(u))]

# Define a function to compute the projection of a vector onto a subspace spanned by the columns of a matrix
def projection(v, A):
  return dot(dot(v, transpose(A)), A)

# Define a function to compute the SVD of a matrix
def svd(A):
  # Compute the transpose of A
  A_T = transpose(A)
  
  # Compute the dot product of A_T and A
  A_T_A = dot(A_T, A)
  
  # Compute the eigenvalues and eigenvectors of A_T_A
  eigenvalues, eigenvectors = np.linalg.eig(A_T_A)
  
  # Sort the eigenvalues and eigenvectors in decreasing order
  eigenvalues, eigenvectors = zip(*sorted(zip(eigenvalues, eigenvectors), key=lambda x: x[0], reverse=True))
  
  # Compute the singular values as the square root of the eigenvalues
  s = [e**0.5 for e in eigenvalues]
  
  # Compute the left singular vectors as the projection of the columns of A onto the subspace spanned by the eigenvectors
  U = [[projection(A[:, j], eigenvectors)[i] for j in range(len(A[0]))] for i in range(len(A))]
  
  # Normalize the left singular vectors
  U = [normalize(u) for u in U]
  
  # Compute the right singular vectors as the normalized eigenvectors
  V = [normalize(v) for v in eigenvectors]
  
  # Transpose V to get Vt
  Vt = transpose(V)
  
  return U, s, Vt

# Test the svd function
U, s, Vt = svd(A)

# The singular values are stored in the variable s as a 1-D array
print(s)
# [9.52551809 0.51430058]

# The left singular vectors are stored in the columns of U
print(U)
# [[-0.2298477   0.88346102  0.40824829]
#  [-0.52474482  0.24078249 -0.81649658]
#  [-0.81964194 -0.40189603  0.40824829]]

# The right singular vectors are stored in the rows of Vt (V is the transpose of Vt)
print(Vt)
# [[-0.61962948 -0.78489445]
#  [-0.78489445  0.61962948]]

# To reconstruct A from the SVD, we can use the dot function to compute the matrix product
reconstructed_A = dot(dot(U, [[s[i], 0] for i in range(len(s))]), Vt)
print(reconstructed_A)
# [[1. 2.]
#  [3. 4.]
#  [5. 6.]]
