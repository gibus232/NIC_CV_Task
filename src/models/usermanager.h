#pragma once

#include <QObject>
#include <QVariantList>
#include <QVariantMap>
#include "../database/database.h"

class UserManager : public QObject
{
    Q_OBJECT
    Q_PROPERTY(int     currentUserId   READ currentUserId   NOTIFY currentUserChanged)
    Q_PROPERTY(QString currentUserName READ currentUserName NOTIFY currentUserChanged)
    Q_PROPERTY(QString currentUserColor READ currentUserColor NOTIFY currentUserChanged)
    Q_PROPERTY(QString currentUserAvatar READ currentUserAvatar NOTIFY currentUserChanged)
    Q_PROPERTY(bool    isLoggedIn      READ isLoggedIn      NOTIFY currentUserChanged)

public:
    explicit UserManager(Database *db, QObject *parent = nullptr);

    int     currentUserId()    const { return m_currentUserId; }
    QString currentUserName()  const { return m_currentUser.value("name").toString(); }
    QString currentUserColor() const { return m_currentUser.value("color", QStringLiteral("#1E90FF")).toString(); }
    QString currentUserAvatar()const { return m_currentUser.value("avatar", QStringLiteral("person")).toString(); }
    bool    isLoggedIn()       const { return m_currentUserId > 0; }

    Q_INVOKABLE QVariantList getUsers();
    Q_INVOKABLE int  createUser(const QString &name, const QString &pin,
                                const QString &color, const QString &avatar);
    Q_INVOKABLE bool updateUser(int id, const QString &name, const QString &pin,
                                const QString &color, const QString &avatar);
    Q_INVOKABLE bool deleteUser(int id);
    Q_INVOKABLE bool login(int userId, const QString &pin);
    Q_INVOKABLE void logout();
    Q_INVOKABLE bool hasPin(int userId);
    Q_INVOKABLE QVariantMap getUserById(int id);

signals:
    void currentUserChanged();
    void usersChanged();

private:
    Database    *m_db;
    int          m_currentUserId = -1;
    QVariantMap  m_currentUser;
    void refreshCurrentUser();
};
