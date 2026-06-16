import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import FamilyBudget

Item {
    id: root

    // Home tab content
    Component {
        id: homeComp

        Flickable {
            id: homeFlic
            contentHeight: homeCol.height + 30
            clip: true

            property string month: budgetManager.currentMonth()
            property double income:   budgetManager.getTotalIncome(userManager.currentUserId, month)
            property double expenses: budgetManager.getTotalExpenses(userManager.currentUserId, month)
            property double balance:  income - expenses

            function refresh() {
                income   = budgetManager.getTotalIncome(userManager.currentUserId, month)
                expenses = budgetManager.getTotalExpenses(userManager.currentUserId, month)
                balance  = income - expenses
            }

            Connections { target: budgetManager; function onDataChanged() { homeFlic.refresh() } }

            Column {
                id: homeCol
                width: parent.width
                spacing: 0

                // Header
                Rectangle {
                    width: parent.width; height: 64
                    color: Qt.rgba(0,0.03,0.12,0.80)

                    Rectangle { anchors.bottom: parent.bottom; anchors.left: parent.left; anchors.right: parent.right; height: 1; color: Qt.rgba(1,1,1,0.18) }

                    Row {
                        anchors.left: parent.left; anchors.leftMargin: 16; anchors.verticalCenter: parent.verticalCenter; spacing: 12

                        Rectangle {
                            width: 40; height: 40; radius: 20
                            gradient: Gradient {
                                GradientStop { position: 0.0; color: Qt.lighter(userManager.currentUserColor, 1.4) }
                                GradientStop { position: 1.0; color: userManager.currentUserColor }
                            }
                            border.color: Qt.rgba(1,1,1,0.45); border.width: 2
                            Text { anchors.centerIn: parent; text: userManager.currentUserName.length > 0 ? userManager.currentUserName[0].toUpperCase() : "?"; color: "white"; font.pixelSize: 18; font.weight: Font.Bold }
                        }

                        Column {
                            anchors.verticalCenter: parent.verticalCenter; spacing: 2
                            Text { text: "Привет, " + userManager.currentUserName; color: "white"; font.pixelSize: 16; font.weight: Font.Medium }
                            Text { text: budgetManager.monthDisplayName(homeFlic.month); color: Qt.rgba(1,1,1,0.58); font.pixelSize: 12 }
                        }
                    }

                    Rectangle {
                        anchors.right: parent.right; anchors.rightMargin: 12; anchors.verticalCenter: parent.verticalCenter
                        width: 40; height: 40; radius: 20
                        color: Qt.rgba(1,1,1,0.10); border.color: Qt.rgba(1,1,1,0.20); border.width: 1
                        Text { anchors.centerIn: parent; text: "⏎"; color: "white"; font.pixelSize: 20 }
                        MouseArea { anchors.fill: parent; onClicked: userManager.logout() }
                    }
                }

                // Balance card
                Rectangle {
                    width: parent.width - 32; height: 130
                    anchors.horizontalCenter: parent.horizontalCenter
                    radius: 22
                    gradient: Gradient {
                        orientation: Gradient.Horizontal
                        GradientStop { position: 0.0; color: "#0D47A1" }
                        GradientStop { position: 1.0; color: "#1565C0" }
                    }
                    border.color: Qt.rgba(1,1,1,0.28); border.width: 1
                    anchors.topMargin: 16

                    Rectangle {
                        anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right; anchors.margins: 1
                        height: parent.height * 0.45; radius: parent.radius
                        gradient: Gradient {
                            GradientStop { position: 0.0; color: Qt.rgba(1,1,1,0.18) }
                            GradientStop { position: 1.0; color: Qt.rgba(1,1,1,0.00) }
                        }
                    }

                    Column {
                        anchors.centerIn: parent; spacing: 8
                        Text { anchors.horizontalCenter: parent.horizontalCenter; text: "Баланс за месяц"; color: Qt.rgba(1,1,1,0.72); font.pixelSize: 14 }
                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: (homeFlic.balance >= 0 ? "+" : "") +
                                  Number(homeFlic.balance).toLocaleString(Qt.locale("ru_RU"),"f",2) + " ₽"
                            color: "white"; font.pixelSize: 36; font.weight: Font.Bold
                        }
                    }
                }

                // Income / Expenses row
                Row {
                    width: parent.width - 32; anchors.horizontalCenter: parent.horizontalCenter
                    spacing: 12

                    Rectangle {
                        width: (parent.width - 12) / 2; height: 80; radius: 18
                        color: Qt.rgba(0.1,0.45,0.1,0.22)
                        border.color: Qt.rgba(0.4,0.9,0.4,0.25); border.width: 1

                        Column {
                            anchors.centerIn: parent; spacing: 6
                            Text { anchors.horizontalCenter: parent.horizontalCenter; text: "↑ Доход"; color: "#A5D6A7"; font.pixelSize: 13 }
                            Text { anchors.horizontalCenter: parent.horizontalCenter; text: Number(homeFlic.income).toLocaleString(Qt.locale("ru_RU"),"f",0)+" ₽"; color: "#66BB6A"; font.pixelSize: 20; font.weight: Font.SemiBold }
                        }
                    }

                    Rectangle {
                        width: (parent.width - 12) / 2; height: 80; radius: 18
                        color: Qt.rgba(0.45,0.1,0.1,0.22)
                        border.color: Qt.rgba(0.9,0.35,0.35,0.25); border.width: 1

                        Column {
                            anchors.centerIn: parent; spacing: 6
                            Text { anchors.horizontalCenter: parent.horizontalCenter; text: "↓ Расход"; color: "#EF9A9A"; font.pixelSize: 13 }
                            Text { anchors.horizontalCenter: parent.horizontalCenter; text: Number(homeFlic.expenses).toLocaleString(Qt.locale("ru_RU"),"f",0)+" ₽"; color: "#EF5350"; font.pixelSize: 20; font.weight: Font.SemiBold }
                        }
                    }
                }

                // Recent transactions
                Text {
                    text: "Последние операции"
                    color: "white"; font.pixelSize: 17; font.weight: Font.Medium
                    anchors.left: parent.left; anchors.leftMargin: 20
                }

                Column {
                    width: parent.width - 32; anchors.horizontalCenter: parent.horizontalCenter; spacing: 10

                    Repeater {
                        model: budgetManager.getRecentTransactions(userManager.currentUserId, 8)
                        delegate: TransactionItem {
                            width: parent.width; tx: modelData
                            onDeleteRequested: function(id) {
                                budgetManager.deleteTransaction(id)
                            }
                        }
                    }

                    Rectangle {
                        visible: budgetManager.getRecentTransactions(userManager.currentUserId, 1).length === 0
                        width: parent.width; height: 80; radius: 16
                        color: Qt.rgba(1,1,1,0.05); border.color: Qt.rgba(1,1,1,0.12); border.width: 1
                        Text { anchors.centerIn: parent; text: "Пока нет операций. Нажмите + чтобы добавить."; color: Qt.rgba(1,1,1,0.40); font.pixelSize: 14 }
                    }
                }
            }
        }
    }

    Component { id: txComp;     TransactionsScreen  {} }
    Component { id: budgetComp; BudgetScreen        {} }
    Component { id: reportsComp; ReportsScreen      {} }
    Component { id: familyComp; FamilyScreen        {} }
    Component { id: addTxComp;  AddTransactionScreen {} }

    // Tab content loader
    StackLayout {
        id: tabs
        anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right; anchors.bottom: bottomNav.top
        currentIndex: bottomNav.currentIndex

        Loader { active: tabs.currentIndex === 0; sourceComponent: homeComp }
        Loader { active: tabs.currentIndex === 1; sourceComponent: txComp }
        Loader { active: tabs.currentIndex === 2; sourceComponent: budgetComp }
        Loader { active: tabs.currentIndex === 3; sourceComponent: reportsComp }
        Loader { active: tabs.currentIndex === 4; sourceComponent: familyComp }
    }

    // FAB – add transaction
    Rectangle {
        anchors.bottom: bottomNav.top; anchors.bottomMargin: 16; anchors.right: parent.right; anchors.rightMargin: 20
        width: 56; height: 56; radius: 28
        visible: bottomNav.currentIndex === 0 || bottomNav.currentIndex === 1
        gradient: Gradient {
            GradientStop { position: 0.0; color: "#42A5F5" }
            GradientStop { position: 1.0; color: "#1565C0" }
        }
        border.color: Qt.rgba(1,1,1,0.45); border.width: 2

        Rectangle {
            anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right; anchors.margins: 1
            height: parent.height * 0.48; radius: parent.radius
            gradient: Gradient {
                GradientStop { position: 0.0; color: Qt.rgba(1,1,1,0.30) }
                GradientStop { position: 1.0; color: Qt.rgba(1,1,1,0.00) }
            }
        }

        Text { anchors.centerIn: parent; text: "+"; color: "white"; font.pixelSize: 30; font.weight: Font.Light }

        scale: fabMa.pressed ? 0.90 : 1.0
        Behavior on scale { NumberAnimation { duration: 80 } }

        MouseArea {
            id: fabMa; anchors.fill: parent
            onClicked: StackView.view.push(addTxComp)
        }
    }

    BottomNav {
        id: bottomNav
        anchors.bottom: parent.bottom; anchors.left: parent.left; anchors.right: parent.right
        onTabChanged: function(i) { currentIndex = i }
    }
}
