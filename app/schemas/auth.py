"""Auth request/response schemas. Fields are plain str so ALL validation errors
go through the endpoint as structured {detail:{code,...}} the SPA already parses."""

from __future__ import annotations

from pydantic import BaseModel


class SignupRequest(BaseModel):
    email: str
    password: str
    name: str


class LoginRequest(BaseModel):
    email: str
    password: str


class UpdateProfileRequest(BaseModel):
    name: str


class ChangePasswordRequest(BaseModel):
    currentPassword: str
    newPassword: str


class MeResponse(BaseModel):
    id: str
    email: str
    name: str
    role: str
    createdAt: str
    authProvider: str = "password"
