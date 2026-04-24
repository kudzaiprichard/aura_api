import bcrypt

from src.configs import security


def hash_password(plain: str) -> str:
    rounds = security.password.bcrypt_rounds
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=rounds)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())
