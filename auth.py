from flask import Blueprint, render_template, request, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user
from models import db, User

auth = Blueprint("auth", __name__)

@auth.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"]).first()
        if user and check_password_hash(user.password, request.form["password"]):
            login_user(user)
            return redirect("/")
    return render_template("login.html")

@auth.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        hashed_pw = generate_password_hash(request.form["password"])
        user = User(username=request.form["username"], password=hashed_pw)
        db.session.add(user)
        db.session.commit()
        return redirect("/login")
    return render_template("register.html")

@auth.route("/logout")
def logout():
    logout_user()
    return redirect("/login")
