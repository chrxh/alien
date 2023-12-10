<?php
    require './helpers.php';

    $db = connectToDB();
    $db->begin_transaction();

    $userName = $_POST["userName"];
    $pw = $_POST["password"];

    if (!checkPw($db, $userName, $pw)) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    $userObj = $db->query("SELECT ID as id FROM user WHERE NAME ='".addslashes($userName)."'")->fetch_object();
    if (!$userObj) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    $db->query("DELETE FROM userlike WHERE USER_ID={$userObj->id}");
    $db->query("DELETE FROM simulation WHERE USER_ID={$userObj->id}");
    $db->query("DELETE FROM user WHERE ID={$userObj->id}");

    echo json_encode(["result"=>true]);

    $db->commit();
    $db->close();
?>