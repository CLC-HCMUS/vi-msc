<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Nén đa câu</title>
    <link rel="stylesheet" type="text/css" href="./Boostrap v4.1/css/bootstrap.min.css">
    <link rel="stylesheet" type="text/css" href="./style.css">
    <script type="text/javascript" src="./jquery-3.3.1.min.js"></script>
    <script type="text/javascript" src="./Boostrap v4.1/js/bootstrap.min.js"></script>
</head>
<?php
$path 	 = "/var/www/html/mds/";
$in 	 = $path . "msc-in";
$out 	 = $path . "msc-out";
$default = $path . "msc-default";
$result  = "";

if (isset($_POST['VanBan'])) {
    file_put_contents($in, trim($_POST['VanBan']));
    exec("cd {$path} && python3 mds.py msc {$in} {$out}");
    $result = file_get_contents($out);
}
?>
<body>
    <div class="TopHeader">
        <div class="wrap">
            <h1 class="Title">CÔNG CỤ NÉN ĐA CÂU</h1>
        </div>
    </div>
    <div class="wrap">
        <div class="FormNoiDung">
            <form action="" method="POST">
                <div class="form-group">
                    <label for="IDVanBan">Nhập các câu cùng chủ đề</label>
                    <textarea name="VanBan" class="form-control" id="IDVanBan" rows="8"><?php echo isset($_POST['VanBan']) ? $_POST['VanBan'] : file_get_contents($default); ?></textarea>
                </div>
                <button type="submit" class="btn btn-primary">Nén câu</button>
            </form>
        </div>

        <div class="FormTomTat">
            <label for="TomTat">Nội dung câu nén</label>
            <textarea class="form-control" id="TomTat" rows="8"><?php echo $result; ?></textarea>
        </div>
    </div>
</body>
</html>
