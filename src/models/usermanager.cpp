#include "usermanager.h"

UserManager::UserManager(Database *db, QObject *parent)
    : QObject(parent), m_db(db) {}

void UserManager::refreshCurrentUser()
{
    m_currentUser = m_currentUserId > 0 ? m_db->getUserById(m_currentUserId) : QVariantMap{};
}

QVariantList UserManager::getUsers()
{
    QVariantList r;
    for (auto &m : m_db->getAllUsers()) r << m;
    return r;
}

int UserManager::createUser(const QString &name, const QString &pin,
                             const QString &color, const QString &avatar)
{
    int id = m_db->createUser(name, pin, color, avatar);
    if (id > 0) emit usersChanged();
    return id;
}

bool UserManager::updateUser(int id, const QString &name, const QString &pin,
                              const QString &color, const QString &avatar)
{
    bool ok = m_db->updateUser(id, name, pin, color, avatar);
    if (ok) {
        emit usersChanged();
        if (id == m_currentUserId) { refreshCurrentUser(); emit currentUserChanged(); }
    }
    return ok;
}

bool UserManager::deleteUser(int id)
{
    bool ok = m_db->deleteUser(id);
    if (ok) {
        emit usersChanged();
        if (id == m_currentUserId) logout();
    }
    return ok;
}

bool UserManager::login(int userId, const QString &pin)
{
    if (!m_db->verifyPin(userId, pin)) return false;
    m_currentUserId = userId;
    refreshCurrentUser();
    emit currentUserChanged();
    return true;
}

void UserManager::logout()
{
    m_currentUserId = -1;
    m_currentUser.clear();
    emit currentUserChanged();
}

bool UserManager::hasPin(int userId)
{
    return m_db->userHasPin(userId);
}

QVariantMap UserManager::getUserById(int id)
{
    return m_db->getUserById(id);
}
