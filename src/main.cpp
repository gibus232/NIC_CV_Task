#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickStyle>
#include <QQmlError>
#include <QDebug>
#include "database/database.h"
#include "models/usermanager.h"
#include "models/budgetmanager.h"

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);
    app.setOrganizationName("FamilyBudget");
    app.setOrganizationDomain("familybudget.app");
    app.setApplicationName("FamilyBudget");
    app.setApplicationVersion("1.0.0");

    QQuickStyle::setStyle("Basic");

    Database db;
    // A DB failure must not take the whole app down at startup – log it and
    // keep going. The managers degrade gracefully (queries just return empty).
    if (!db.initialize())
        qWarning() << "[FamilyBudget] Database initialization failed; continuing with limited data.";

    UserManager userManager(&db);
    BudgetManager budgetManager(&db);

    QQmlApplicationEngine engine;
    engine.rootContext()->setContextProperty("userManager", &userManager);
    engine.rootContext()->setContextProperty("budgetManager", &budgetManager);

    // Surface every QML warning to logcat so issues are diagnosable instead of
    // silently killing the process.
    QObject::connect(&engine, &QQmlApplicationEngine::warnings,
                     [](const QList<QQmlError> &warnings) {
                         for (const QQmlError &e : warnings)
                             qWarning() << "[QML]" << e.toString();
                     });

    const QUrl url(QStringLiteral("qrc:/qt/qml/FamilyBudget/qml/main.qml"));
    QObject::connect(
        &engine, &QQmlApplicationEngine::objectCreated, &app,
        [url](QObject *obj, const QUrl &objUrl) {
            if (!obj && url == objUrl)
                qCritical() << "[FamilyBudget] Failed to create root object from" << objUrl.toString();
        },
        Qt::QueuedConnection);

    engine.load(url);

    if (engine.rootObjects().isEmpty())
        qCritical() << "[FamilyBudget] No root QML objects were created.";

    return app.exec();
}
