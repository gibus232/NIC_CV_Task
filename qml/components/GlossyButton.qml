import QtQuick

Rectangle {
    id: root
    property string label: ""
    property color baseColor: "#1565C0"
    property bool enabled: true
    signal clicked()

    height: 52
    radius: height / 2
    opacity: root.enabled ? 1.0 : 0.45

    gradient: Gradient {
        GradientStop { position: 0.0; color: Qt.lighter(root.baseColor, 1.35) }
        GradientStop { position: 0.5; color: root.baseColor }
        GradientStop { position: 1.0; color: Qt.darker(root.baseColor, 1.30) }
    }
    border.color: Qt.rgba(1, 1, 1, 0.35)
    border.width: 1

    Rectangle {
        anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right; anchors.topMargin: 1; anchors.leftMargin: 1; anchors.rightMargin: 1
        height: parent.height * 0.48
        radius: parent.radius
        gradient: Gradient {
            GradientStop { position: 0.0; color: Qt.rgba(1, 1, 1, 0.28) }
            GradientStop { position: 1.0; color: Qt.rgba(1, 1, 1, 0.00) }
        }
    }

    Text {
        anchors.centerIn: parent
        text: root.label
        color: "white"
        font.pixelSize: 16
        font.weight: Font.Medium
    }

    scale: area.pressed ? 0.96 : 1.0
    Behavior on scale { NumberAnimation { duration: 80 } }

    MouseArea {
        id: area
        anchors.fill: parent
        enabled: root.enabled
        onClicked: root.clicked()
    }
}
