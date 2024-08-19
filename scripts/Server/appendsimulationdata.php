<?php
    require './helpers.php';
    require './hooks.php';

    $db = connectToDB();
    $db->begin_transaction();

    $userName = $_POST["userName"];
    $pw = $_POST["password"];
    $chunkIndex = (int)$_POST["chunkIndex"];

    if (!checkPw($db, $userName, $pw)) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    $userObj = $db->query("SELECT u.ID as id FROM user u WHERE u.NAME='".addslashes($userName)."'")->fetch_object();
    if (!$userObj) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    $simId = $_POST['simId'];
    $content = $_POST['content'];
    $size = strlen($content);

    $obj = $db->query("SELECT sim.USER_ID as userId, sim.SIZE as size FROM simulation sim WHERE sim.ID=$simId")->fetch_object();
    if (!$obj || strcmp($userObj->id, $obj->userId) != 0 ) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    $newSize = (int)$obj->size + $size;

    if (!$db->query("UPDATE simulation SET TIMESTAMP=TIMESTAMP, content" . (string)($chunkIndex + 1) . " = '" . addslashes($content) . "', size = $newSize WHERE ID = $simId")) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    echo json_encode(["result"=>true]);

    $db->commit();
    $db->close();
?>