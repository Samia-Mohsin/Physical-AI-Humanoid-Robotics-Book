import bcrypt

def test_direct_bcrypt(password: str) -> str:
    """Test function to hash a password using direct bcrypt"""
    print(f"Original password length: {len(password)}")
    print(f"Original password bytes length: {len(password.encode('utf-8'))}")

    # Truncate password to 72 bytes to comply with bcrypt limits
    if len(password) > 72:
        password = password[:72]
        print(f"Truncated password length: {len(password)}")

    # Convert password to bytes
    password_bytes = password.encode('utf-8')

    # Hash the password
    hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
    return hashed.decode('utf-8')

# Test with a simple password
try:
    hashed = test_direct_bcrypt("simplepass")
    print(f"Successfully hashed password: {hashed}")
except Exception as e:
    print(f"Error hashing password: {e}")
    import traceback
    traceback.print_exc()

# Test with a longer password
try:
    long_password = "a" * 80  # 80 character password
    print(f"\nTesting with long password (length: {len(long_password)})")
    hashed = test_direct_bcrypt(long_password)
    print(f"Successfully hashed long password: {hashed}")
except Exception as e:
    print(f"Error hashing long password: {e}")
    import traceback
    traceback.print_exc()