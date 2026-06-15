import QtQuick

Rectangle {
    id: root
    property string title: ""
    property bool showBack: false
    property bool showAction: false
    property string actionIcon: "+"
    signal backClicked()
    signal actionClicked()

    height: 60
    color: Qt.rgba(0, 0.03, 0.12, 0.80)

    Rectangle {
        anchors { bottom: parent.bottom; left: parent.left; right: parent.right }
        height: 1
        color: Qt.rgba(1, 1, 1, 0.18)
    }

    Rectangle {
        visible: root.showBack
        anchors { left: parent.left; leftMargin: 10; verticalCenter: parent.verticalCenter }
        width: 42; height: 42; radius: 21
        color: Qt.rgba(1, 1, 1, 0.10)
        border.color: Qt.rgba(1,1,1,0.18); border.width: 1

        Text { anchors.centerIn: parent; text: "‹"; color: "white"; font.pixelSize: 28 }
        MouseArea { anchors.fill: parent; onClicked: root.backClicked() }
    }

    Text {
        anchors.centerIn: parent
        text: root.title
        color: "white"
        font.pixelSize: 19
        font.weight: Font.Medium
    }

    Rectangle {
        visible: root.showAction
        anchors { right: parent.right; rightMargin: 10; verticalCenter: parent.verticalCenter }
        width: 42; height: 42; radius: 21
        color: Qt.rgba(1, 1, 1, 0.10)
        border.color: Qt.rgba(1,1,1,0.18); border.width: 1

        Text { anchors.centerIn: parent; text: root.actionIcon; color: "white"; font.pixelSize: 22 }
        MouseArea { anchors.fill: parent; onClicked: root.actionClicked() }
    }
}
