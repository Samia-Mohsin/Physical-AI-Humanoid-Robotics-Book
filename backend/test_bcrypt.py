from passlib.context import CryptContext

# Test bcrypt functionality
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def test_password_hash(password: str) -> str:
    """Test function to hash a password"""
    print(f"Original password length: {len(password)}")
    print(f"Original password bytes length: {len(password.encode('utf-8'))}")

    # Truncate password to 72 bytes to comply with bcrypt limits
    if len(password) > 72:
        password = password[:72]
        print(f"Truncated password length: {len(password)}")

    return pwd_context.hash(password)

# Test with a simple password
try:
    hashed = test_password_hash("simplepass")
    print(f"Successfully hashed password: {hashed}")
except Exception as e:
    print(f"Error hashing password: {e}")
    import traceback
    traceback.print_exc()

# Test with a longer password
try:
    long_password = "a" * 80  # 80 character password
    print(f"\nTesting with long password (length: {len(long_password)})")
    hashed = test_password_hash(long_password)
    print(f"Successfully hashed long password: {hashed}")
except Exception as e:
    print(f"Error hashing long password: {e}")
    import traceback
    traceback.print_exc()